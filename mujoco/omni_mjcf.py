#!/usr/bin/env python3
"""Parametric MJCF generator for N-wheel omnidirectional robots.

Rebuilds the wheel model used by the Gazebo URDFs in ../urdf (3w_v2, 4w, 5w, 6w)
in MuJoCo, with every dimension exposed so it can be scaled down (e.g. to an
IEEE VSSS robot):

  * each omni wheel = hub on a driven hinge + `rows` x `rollers_per_row` passive
    rollers on free hinges (the Gazebo models use 2 rows x 3 rollers),
  * rollers are barrels whose surface follows the wheel's rolling circle (same idea
    as ../meshes/4w/roller.stl), so the wheel stays round. Default is an ellipsoid
    with its length tuned to minimise the bump (smooth analytic contact, no
    faceting vibration); roller_shape="mesh" gives the exact profile as a mesh,
  * wheel i sits at angle heading_offset + 360*i/N, spin axis pointing inward
    (same sign convention as src/kinematics.cpp: +speed -> CCW tangent motion).

MuJoCo convex-hulls every mesh, so the repo's non-convex chassis/hub STLs can't be
reused as collision geometry; the chassis here is a box and the hub is visual only.

Usage:
  python omni_mjcf.py                  # writes vsss_omni4.xml with the defaults
  python omni_mjcf.py --motor dc -o my_robot.xml
  python -m mujoco.viewer --mjcf=vsss_omni4.xml
"""
import argparse
import math
import warnings
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class OmniParams:
    # --- layout -----------------------------------------------------------------
    n_wheels: int = 4
    heading_offset_deg: float = -45.0  # -45 with 4 wheels = "X" layout (repo 4w); 0 = "+" layout
    wheel_R: float = 0.028  # chassis center -> wheel mid-plane (the R in the kinematics)
    # --- chassis (collision box, bottom sits `ground_clearance` above the floor) --
    body_size: float = 0.075  # side of the square footprint [m]
    body_height: float = 0.070  # top of chassis above the floor [m]
    ground_clearance: float = 0.004
    body_mass: float = 0.150  # everything except wheels [kg]
    com_height: float = 0.020  # chassis centre of mass above the floor [m] (battery low = small)
    # --- omni wheel ----------------------------------------------------------------
    wheel_radius: float = 0.016  # rolling radius (the r in the kinematics) [m]
    roller_shape: str = "ellipsoid"  # "ellipsoid" (smooth, fast) or "mesh" (exact profile, faceted)
    rows: int = 2  # roller rows per wheel (1 or 2)
    rollers_per_row: int = 3
    row_spacing: float = 0.006  # axial distance between the two row planes [m]
    roller_radius: float = 0.005  # roller radius at its middle [m]
    roller_overlap_deg: float = 4.0  # mesh rollers: extra angular coverage so rows overlap
    roller_half_length: float = 0.0  # half the roller length [m]; 0 = auto (see roller_half_length())
    roller_end_gap: float = 0.001  # min clearance between roller ends in one row (spokes/pins) [m]
    hub_mass: float = 0.008  # [kg] per wheel
    roller_mass: float = 0.0005  # [kg] per roller
    roller_damping: float = 1e-7  # bearing friction of rollers [N m s/rad]
    friction: float = 0.8  # roller/floor sliding friction
    contact_timeconst: float = 0.01  # roller contact softness [s] (>= 2*timestep); stiffer -> chatter at high rpm
    # --- motor ---------------------------------------------------------------------
    motor: str = "servo"  # "servo": ideal-ish speed loop (like the Gazebo sim); "dc": voltage-driven DC motor
    servo_kv: float = 0.05  # speed-loop gain [N m / (rad/s)]
    max_wheel_speed: float = 60.0  # servo ctrl range [rad/s]
    stall_torque: float = 0.10  # at the wheel (after gearbox) [N m]; also servo torque limit
    no_load_speed: float = 62.8  # at the wheel, at nominal_voltage [rad/s]
    nominal_voltage: float = 6.0
    wheel_armature: float = 1e-5  # reflected rotor inertia J_rotor * gear^2 [kg m^2]
    # --- simulation ------------------------------------------------------------
    timestep: float = 0.0005


def wheel_angles(p: OmniParams) -> np.ndarray:
    return np.radians(p.heading_offset_deg + 360.0 * np.arange(p.n_wheels) / p.n_wheels)


def wheel_jacobian(p: OmniParams, R=None, r=None) -> np.ndarray:
    """N x 3 matrix mapping body twist (vx, vy, wz) to wheel speeds [rad/s].

    Same matrix as OmniKinematics::init_transform_matrix in src/kinematics.cpp.
    """
    R = p.wheel_R if R is None else R
    r = p.wheel_radius if r is None else r
    th = wheel_angles(p)
    return np.stack([-np.sin(th), np.cos(th), np.full_like(th, R)], axis=1) / r


def wheel_speeds(p: OmniParams, vx, vy, wz) -> np.ndarray:
    return wheel_jacobian(p) @ np.array([vx, vy, wz])


def body_twist(p: OmniParams, w) -> np.ndarray:
    """Forward kinematics (odometry): least-squares twist from N wheel speeds."""
    return np.linalg.pinv(wheel_jacobian(p)) @ np.asarray(w)


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _roller_outline(p: OmniParams, shape, half_len, n=61):
    """Roller cross-section in the wheel plane as (radial, axial) points around its centre."""
    rho0 = p.roller_radius
    if shape == "ellipsoid":
        t = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        return np.stack([rho0 * np.cos(t), half_len * np.sin(t)], 1)
    r, d = p.wheel_radius, p.wheel_radius - rho0
    s = np.linspace(-half_len, half_len, n)
    rho = np.sqrt(r * r - s * s) - d
    cap = np.linspace(-rho[0], rho[0], 11)
    return np.concatenate([
        np.stack([rho, s], 1), np.stack([-rho, s], 1),
        np.stack([cap, np.full_like(cap, s[0])], 1), np.stack([cap, np.full_like(cap, s[-1])], 1),
    ])


def _place(p: OmniParams, outline, phi):
    """Outline of a roller centred at wheel angle phi, in wheel-plane coordinates."""
    er, et = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
    return (p.wheel_radius - p.roller_radius) * er + outline[:, :1] * er + outline[:, 1:] * et


def max_roller_half_length(p: OmniParams, shape):
    """Longest roller that leaves roller_end_gap to its neighbour in the same row.

    Matters for single-row (thin) wheels, where neighbours sit only 360/n degrees apart.
    """
    r, d = p.wheel_radius, p.wheel_radius - p.roller_radius
    hi = 0.98 * math.sqrt(r * r - d * d) if shape == "mesh" else 3 * math.sqrt(p.roller_radius * r)
    gap = max(p.roller_end_gap, 1e-5)

    def clear(a):
        A = _place(p, _roller_outline(p, shape, a), 0.0)
        B = _place(p, _roller_outline(p, shape, a), 2 * math.pi / p.rollers_per_row)
        return np.min(np.linalg.norm(A[:, None] - B[None], axis=2)) >= gap

    if clear(hi):
        return hi
    lo = 1e-5
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if clear(mid) else (lo, mid)
    return lo


def wheel_envelope(p: OmniParams, shape, half_len, n=1441):
    """Rolling radius vs. wheel angle (the wheel's 'roundness') for the given rollers."""
    outline = _roller_outline(p, shape, half_len)
    pts = np.concatenate([
        _place(p, outline, 2 * np.pi * j / p.rollers_per_row + k * np.pi / p.rollers_per_row)
        for k in range(p.rows) for j in range(p.rollers_per_row)
    ])
    psi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (pts @ np.stack([np.cos(psi), np.sin(psi)])).max(0)


def roller_half_length(p: OmniParams):
    """Semi-length of the rollers actually generated, plus the rolling radius range.

    Ellipsoid: sqrt(rho0 * r) matches the wheel curvature at the roller middle; a slightly
    longer roller flattens the dips where contact hands over, so the length is tuned for
    roundness, capped so rollers in one row don't overlap. Mesh: the exact profile over
    180/(rows*n) + roller_overlap_deg, with the same cap. roller_half_length > 0 overrides.
    Returns (half_len, env_min, env_max, max_half_len).
    """
    shape = p.roller_shape
    a_max = max_roller_half_length(p, shape)
    if p.roller_half_length > 0:
        a = p.roller_half_length
    elif shape == "mesh":
        alpha = math.radians(180.0 / (p.rows * p.rollers_per_row) + p.roller_overlap_deg)
        a = min(p.wheel_radius * math.sin(alpha), a_max)
    else:
        a0 = math.sqrt(p.roller_radius * p.wheel_radius)
        hi = min(1.6 * a0, a_max)
        a = min(np.linspace(min(a0, hi), hi, 61), key=lambda a: np.ptp(wheel_envelope(p, shape, a)))
    env = wheel_envelope(p, shape, a)
    return a, env.min(), env.max(), a_max


def _roller_mesh(p: OmniParams, half_len, n_s=17, n_phi=32):
    """Barrel whose surface lies on the wheel's rolling circle.

    A roller whose axis is tangent to the wheel at distance d = r - rho0 from the
    hub has, at axial position s, radius rho(s) = sqrt(r^2 - s^2) - d.
    """
    r, d = p.wheel_radius, p.wheel_radius - p.roller_radius
    verts = []
    for s in np.linspace(-half_len, half_len, n_s):
        rho = math.sqrt(r * r - s * s) - d
        for a in np.linspace(0, 2 * math.pi, n_phi, endpoint=False):
            verts += [rho * math.cos(a), rho * math.sin(a), s]
    return verts


def build_mjcf(p: OmniParams) -> str:
    if p.motor not in ("servo", "dc"):
        raise ValueError("motor must be 'servo' or 'dc'")
    if p.roller_shape not in ("ellipsoid", "mesh"):
        raise ValueError("roller_shape must be 'ellipsoid' or 'mesh'")
    if p.rows not in (1, 2):
        raise ValueError("rows must be 1 or 2")
    r = p.wheel_radius
    d = r - p.roller_radius  # hub centre -> roller axis
    a, _, _, a_max = roller_half_length(p)
    if a > a_max * 1.001:
        warnings.warn(f"roller_half_length {a*1e3:.2f} mm overlaps the neighbouring roller "
                      f"(max {a_max*1e3:.2f} mm with roller_end_gap {p.roller_end_gap*1e3:g} mm)")
    if p.roller_shape == "mesh":
        roller_geom = 'type="mesh" mesh="roller"'
        mesh_asset = f'<mesh name="roller" vertex="{_fmt(_roller_mesh(p, a))}"/>'
    else:
        roller_geom = f'type="ellipsoid" size="{p.roller_radius:g} {p.roller_radius:g} {a:.5g}"'
        mesh_asset = ""
    # Contact bits: floor=1; rollers collide only with the floor/ball (contype 2);
    # chassis (contype 4) collides with floor and other chassis, never own rollers.
    floor = 'contype="1" conaffinity="1"'
    roller = 'contype="2" conaffinity="1"'
    chassis = 'contype="4" conaffinity="5"'
    visual = 'contype="0" conaffinity="0" group="1"'

    half = p.body_size / 2
    z_bot, z_top = p.ground_clearance - r, p.body_height - r  # chassis frame is at axle height
    box_z, box_h = (z_bot + z_top) / 2, (z_top - z_bot) / 2
    # chassis inertia ~ uniform box of the chassis size, but with the COM where you put it
    box_inertia = (
        p.body_mass / 12 * (p.body_size**2 + (2 * box_h) ** 2),
        p.body_mass / 12 * (p.body_size**2 + (2 * box_h) ** 2),
        p.body_mass / 6 * p.body_size**2,
    )

    wheels = []
    row_z = [0.0] if p.rows == 1 else [-p.row_spacing / 2, p.row_spacing / 2]
    for i, th in enumerate(wheel_angles(p), start=1):
        pos = (p.wheel_R * math.cos(th), p.wheel_R * math.sin(th), 0.0)
        axle = (-math.cos(th), -math.sin(th), 0.0)  # inward, like the URDF joints
        rollers = []
        for k, z in enumerate(row_z):
            for j in range(p.rollers_per_row):
                phi = 2 * math.pi * j / p.rollers_per_row + k * math.pi / p.rollers_per_row
                rpos = (d * math.cos(phi), d * math.sin(phi), z)
                raxis = (-math.sin(phi), math.cos(phi), 0.0)  # tangent to the wheel
                rollers.append(
                    f'<body name="roller_{i}_{k}{j}" pos="{_fmt(rpos)}" zaxis="{_fmt(raxis)}">'
                    f'<joint name="roller_{i}_{k}{j}" axis="0 0 1" damping="{p.roller_damping:g}"/>'
                    f'<geom class="roller" {roller_geom} mass="{p.roller_mass:g}" {roller}/>'
                    "</body>"
                )
        wheels.append(
            f'<body name="wheel_{i}" pos="{_fmt(pos)}" zaxis="{_fmt(axle)}">'
            f'<joint name="wheel_{i}" axis="0 0 1" armature="{p.wheel_armature:g}"/>'
            f'<geom type="cylinder" size="{d*0.9:.5g} {p.row_spacing/2 + p.roller_radius*0.6:.5g}" '
            f'mass="{p.hub_mass:g}" rgba=".8 .8 .8 1" {visual}/>'
            + "".join(rollers)
            + "</body>"
        )

    acts = []
    for i in range(1, p.n_wheels + 1):
        if p.motor == "servo":
            acts.append(
                f'<velocity name="motor_{i}" joint="wheel_{i}" kv="{p.servo_kv:g}" '
                f'ctrlrange="{-p.max_wheel_speed:g} {p.max_wheel_speed:g}" '
                f'forcelimited="true" forcerange="{-p.stall_torque:g} {p.stall_torque:g}"/>'
            )
        else:  # tau = (stall/V) * volts - (stall/no_load) * omega
            acts.append(
                f'<general name="motor_{i}" joint="wheel_{i}" '
                f'gainprm="{p.stall_torque/p.nominal_voltage:g}" biastype="affine" '
                f'biasprm="0 0 {-p.stall_torque/p.no_load_speed:g}" '
                f'ctrlrange="{-p.nominal_voltage:g} {p.nominal_voltage:g}"/>'
            )

    sensors = "".join(
        f'<jointvel name="enc_{i}" joint="wheel_{i}"/>' for i in range(1, p.n_wheels + 1)
    )
    return f"""<mujoco model="omni{p.n_wheels}">
  <!-- generated by omni_mjcf.py: {', '.join(f'{f.name}={getattr(p, f.name)}' for f in fields(p))} -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p.timestep:g}" integrator="implicitfast"/>
  <default>
    <default class="roller">
      <geom priority="1" friction="{p.friction:g} 0.005 0.0001" solref="{p.contact_timeconst:g} 1" rgba=".1 .1 .1 1"/>
    </default>
  </default>
  <asset>
    {mesh_asset}
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".15 .15 .15" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" {floor}/>
    <body name="chassis" pos="0 0 {r + 0.001:.5g}">
      <freejoint name="root"/>
      <site name="imu"/>
      <inertial pos="0 0 {p.com_height - r:.5g}" mass="{p.body_mass:g}" diaginertia="{_fmt(box_inertia)}"/>
      <geom name="chassis" type="box" size="{half:g} {half:g} {box_h:.5g}" pos="0 0 {box_z:.5g}"
            mass="0" rgba=".2 .4 .9 .6" {chassis}/>
      <geom type="box" size="0.01 0.003 0.001" pos="{half:g} 0 {z_top:.5g}" rgba="1 0 0 1" {visual}/>
      {''.join(wheels)}
    </body>
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    {sensors}
  </sensor>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="vsss_omni4.xml")
    for f in fields(OmniParams):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = vars(ap.parse_args())
    out = a.pop("out")
    p = OmniParams(**a)
    with open(out, "w") as fh:
        fh.write(build_mjcf(p))
    print(f"wrote {out}")
    a, lo, hi, a_max = roller_half_length(p)
    print(f"{p.roller_shape} rollers: half-length {a*1e3:.2f} mm (same-row limit {a_max*1e3:.2f} mm), "
          f"rolling radius {lo*1e3:.3f}..{hi*1e3:.3f} mm (bump {(hi-lo)*1e3:.3f} mm)")
    print("wheel jacobian (rad/s per [vx, vy, wz]):\n", np.round(wheel_jacobian(p), 3))


if __name__ == "__main__":
    main()
