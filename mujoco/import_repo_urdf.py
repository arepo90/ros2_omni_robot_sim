#!/usr/bin/env python3
"""1:1 import of this repo's Gazebo robots (3w_v2, 4w, 5w, 6w) into MuJoCo.

Expands the xacro, loads it with MuJoCo's URDF importer, and adds what Gazebo
got from elsewhere: a free joint, a floor, velocity actuators (stand-in for
ign_ros2_control's velocity interface) and collision filtering.

The filtering is required: MuJoCo replaces every collision mesh with its convex
hull, and the hull of base_link.stl is a solid block that swallows all rollers,
so the robot cannot move until chassis<->roller contacts are disabled.

  pip install mujoco xacro numpy
  python import_repo_urdf.py 4w                # writes repo_4w.xml and runs a drive test
  python -m mujoco.viewer --mjcf=repo_4w.xml
"""
import argparse
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import xacro

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# same tables as src/kinematics.cpp
WHEELS = {"3w_v2": 3, "4w": 4, "5w": 5, "6w": 6}
HEADING_OFFSET = {"3w_v2": 0, "4w": -45, "5w": 0, "6w": 0}
WHEEL_RADIUS = 0.03
ROBOT_RADIUS_CPP = 0.088  # value hard-coded in src/kinematics.cpp
ROBOT_RADIUS_URDF = 0.1028  # hub-to-wheel-mid-plane distance in these URDFs


def urdf_to_mjcf(model: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        # resolve $(find ros2_omni_robot_sim) without needing a sourced ROS workspace
        for root, _, files in os.walk(os.path.join(PKG, "urdf", model)):
            for f in files:
                src = open(os.path.join(root, f)).read().replace("$(find ros2_omni_robot_sim)", tmp)
                dst = os.path.join(tmp, "urdf", model, os.path.relpath(os.path.join(root, f), os.path.join(PKG, "urdf", model)))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "w").write(src)
        urdf = xacro.process_file(os.path.join(tmp, "urdf", model, "main.urdf.xacro")).toxml()

    meshdir = os.path.join(PKG, "meshes", model)
    urdf = re.sub(r"package://ros2_omni_robot_sim/meshes/[^/]+/", "", urdf).replace(".dae", ".stl")  # MuJoCo can't read .dae
    urdf = re.sub(
        r"(<robot[^>]*>)",
        rf'\1<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" discardvisual="true" fusestatic="false"/></mujoco>',
        urdf, count=1,
    )
    m = mujoco.MjModel.from_xml_string(urdf)
    with tempfile.NamedTemporaryFile(suffix=".xml") as f:
        mujoco.mj_saveLastXML(f.name, m)
        root = ET.parse(f.name).getroot()

    ET.SubElement(root, "option", {"timestep": "0.001", "integrator": "implicitfast"})
    wb = root.find("worldbody")
    base = wb.find("body")
    base.set("pos", "0 0 0.02")
    base.insert(0, ET.Element("freejoint", {"name": "root"}))
    for g in root.iter("geom"):  # robot geoms touch only the floor (Gazebo: mu=1 on rollers)
        g.set("contype", "2")
        g.set("conaffinity", "1")
        g.set("friction", "1 0.005 0.0001")
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.1", "contype": "1", "conaffinity": "2"})
    ET.SubElement(wb, "light", {"pos": "0 0 2", "dir": "0 0 -1"})
    act = ET.SubElement(root, "actuator")
    for i in range(1, WHEELS[model] + 1):
        ET.SubElement(act, "velocity", {"name": f"w{i}", "joint": f"omni_wheel_joint_{i}", "kv": "5", "ctrlrange": "-10 10"})
    return ET.tostring(root, encoding="unicode")


def jacobian(model, R):
    n = WHEELS[model]
    th = np.radians(HEADING_OFFSET[model] + 360.0 * np.arange(n) / n)
    return np.stack([-np.sin(th), np.cos(th), np.full(n, R)], 1) / WHEEL_RADIUS


def drive_test(model, xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    print(f"{model}: {m.nbody-1} bodies, mass {m.body_subtreemass[1]:.2f} kg")
    for cmd, R, label in [
        ((0.2, 0, 0), ROBOT_RADIUS_CPP, "vx"),
        ((0, 0.2, 0), ROBOT_RADIUS_CPP, "vy"),
        ((0, 0, 1.0), ROBOT_RADIUS_CPP, "wz, R from kinematics.cpp"),
        ((0, 0, 1.0), ROBOT_RADIUS_URDF, "wz, R from URDF"),
    ]:
        mujoco.mj_resetData(m, d)
        for _ in range(300):
            mujoco.mj_step(m, d)
        d.ctrl[:] = jacobian(model, R) @ np.array(cmd)
        v = []
        for k in range(2000):
            mujoco.mj_step(m, d)
            if k > 1000:
                q = d.qpos[3:7]
                yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                c, s = np.cos(yaw), np.sin(yaw)
                v.append([c * d.qvel[0] + s * d.qvel[1], -s * d.qvel[0] + c * d.qvel[1], d.qvel[5]])
        v = np.mean(v, 0)
        i = int(np.argmax(np.abs(cmd)))
        print(f"  {label:28s} commanded {cmd[i]:.2f} -> got {v[i]:.3f} ({v[i]/cmd[i]*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(WHEELS))
    ap.add_argument("-o", "--out")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    xml = urdf_to_mjcf(a.model)
    out = a.out or f"repo_{a.model}.xml"
    open(out, "w").write(xml)
    print(f"wrote {out}")
    if not a.no_test:
        drive_test(a.model, xml)


if __name__ == "__main__":
    main()
