#!/usr/bin/env python3
"""ROS 2 bridge for the VSSS MuJoCo scene: drive the robots with cmd_vel, watch their state and the
overhead camera. A plain rclpy script, no colcon build needed.

  source /opt/ros/humble/setup.bash
  python3 ros_bridge.py                       # 3 blue robots, the ball, overhead camera, 3D viewer window
  python3 ros_bridge.py --blue 3 --yellow 3   # 3v3
  python3 ros_bridge.py --no_viewer           # headless
  python3 ros_bridge.py --model roller        # detailed roller-level robots (slow; 3v3 below real time)

  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/blue_0/cmd_vel
  ros2 run rqt_image_view rqt_image_view /overhead_camera/image_raw
  ros2 topic echo /blue_0/odom

Topics, per robot (namespace = robot name: blue_0 .. blue_4, yellow_0 .. yellow_4):
  <robot>/cmd_vel   geometry_msgs/Twist         in: body-frame linear.x, linear.y [m/s], angular.z [rad/s]
  <robot>/odom      nav_msgs/Odometry           ground truth: pose in "field", twist in <robot>/base_link
  <robot>/imu       sensor_msgs/Imu             IMU site at the chassis centre (noise-free)
  <robot>/wheels    sensor_msgs/JointState      wheel angle [rad], speed [rad/s], torque on the wheel [N m]
  <robot>/duty      std_msgs/Float64MultiArray  PWM duty per wheel, after the ramp limiter
and
  /ball/odom                      nav_msgs/Odometry  ground truth; orientation not tracked (identity)
  /overhead_camera/image_raw      sensor_msgs/Image (rgb8), at --cam_hz
  /overhead_camera/camera_info    sensor_msgs/CameraInfo (pinhole; not published with --cam_ortho)
  /tf                             field -> <robot>/base_link; static field -> overhead_camera_optical
  /clock                          sim time
  /reset                          std_srvs/Empty: robots and ball back to their start poses

cmd_vel -> wheels does what the firmware would: inverse kinematics with kinematic_radius, then the
duty from the no-load motor curve at the supply voltage (x --ff_gain), scaled down together if a
wheel would pass 100 %, then a ramp limiter (full scale in --ramp seconds). That is open loop, so
the robot reaches ~75-82 % of the commanded speed (see reference.md); close the loop on /odom or
the camera, or raise --ff_gain.

Time stamps (and /clock) are wall-clock start time + sim time, so tools work with or without
use_sim_time as long as the sim keeps up with real time; if it falls behind, set use_sim_time.
"""
import argparse
import array
import math
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from drive import Drive
from omni_mjcf import wheel_jacobian
from planar_robot import PlanarDrive
from vsss_field import add_scene_args, scene_from_args


class DriveList:
    """One drive.Drive per robot (roller model) behind PlanarDrive's interface."""

    def __init__(self, m, p, prefixes):
        self.drives = [Drive(m, p, prefix=pre) for pre in prefixes]

    def reset(self, d):
        for dr in self.drives:
            dr.reset(d)

    def step(self, d, duty):
        for dr, u in zip(self.drives, duty):
            dr.apply(d, u)
        mujoco.mj_step(dr.m, d)
        for dr in self.drives:
            dr.update(d)

    def wheel_state(self, d):
        return (np.array([d.qpos[dr.qadr] for dr in self.drives]), np.array([d.qvel[dr.dof] for dr in self.drives]),
                np.array([d.actuator_force[dr.act] for dr in self.drives]))


class SimRobot:
    """One robot's MuJoCo handles, command state and publishers."""

    def __init__(self, node, m, scene, robot):
        p = scene.p
        self.name, self.p = robot.name, p
        self.body = m.body(f"{robot.name}/chassis").id
        self.sens = {s: (m.sensor(f"{robot.name}/{s}").adr[0], m.sensor(f"{robot.name}/{s}").dim[0])
                     for s in ("imu_quat", "imu_gyro", "imu_acc")}
        self.J = wheel_jacobian(p)
        self.w_per_duty = p.no_load_speed * (p.supply_voltage or p.nominal_voltage) / p.nominal_voltage
        self.cmd = np.zeros(3)
        self.cmd_wall = 0.0
        self.duty = np.zeros(p.n_wheels)
        ns = robot.name
        node.create_subscription(Twist, f"{ns}/cmd_vel", self._on_cmd, 10)
        self.pub_odom = node.create_publisher(Odometry, f"{ns}/odom", 10)
        self.pub_imu = node.create_publisher(Imu, f"{ns}/imu", 10)
        self.pub_wheels = node.create_publisher(JointState, f"{ns}/wheels", 10)
        self.pub_duty = node.create_publisher(Float64MultiArray, f"{ns}/duty", 10)
        self.wheel_names = [f"{ns}/wheel_{i}" for i in range(1, p.n_wheels + 1)]

    def _on_cmd(self, msg: Twist):  # rclpy thread: plain assignments only
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_wall = time.monotonic()

    def update_duty(self, dt, a):
        cmd = self.cmd
        if a.cmd_timeout > 0 and time.monotonic() - self.cmd_wall > a.cmd_timeout:
            cmd = np.zeros(3)
        target = a.ff_gain * (self.J @ cmd) / self.w_per_duty
        peak = np.abs(target).max()
        if peak > 1.0:
            target = target / peak  # keep the direction, slow everything down
        step = dt / a.ramp if a.ramp > 0 else np.inf
        self.duty = self.duty + np.clip(target - self.duty, -step, step)

    def publish(self, m, d, stamp, tfs, wheels):
        pos, quat = d.xpos[self.body], d.xquat[self.body]  # quat: w x y z
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.body, vel, 1)  # body frame: angular, linear
        w_body, v_body = vel[:3], vel[3:]
        base = f"{self.name}/base_link"

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, "field", base
        _set_pose(od.pose.pose, pos, quat)
        _set_vec(od.twist.twist.linear, v_body)
        _set_vec(od.twist.twist.angular, w_body)
        self.pub_odom.publish(od)

        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, base
        q = self._sensor(d, "imu_quat")
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = map(float, q)
        _set_vec(imu.angular_velocity, self._sensor(d, "imu_gyro"))
        _set_vec(imu.linear_acceleration, self._sensor(d, "imu_acc"))
        self.pub_imu.publish(imu)

        js = JointState()
        js.header.stamp = stamp
        js.name = self.wheel_names
        js.position, js.velocity, js.effort = ([float(x) for x in w] for w in wheels)
        self.pub_wheels.publish(js)

        self.pub_duty.publish(Float64MultiArray(data=[float(x) for x in self.duty]))

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "field", base
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
        tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = map(float, quat)
        tfs.append(tf)

    def _sensor(self, d, name):
        adr, dim = self.sens[name]
        return d.sensordata[adr:adr + dim]


def _set_vec(v, x):
    v.x, v.y, v.z = (float(c) for c in x)


def _set_pose(pose, pos, quat_wxyz):
    pose.position.x, pose.position.y, pose.position.z = (float(c) for c in pos)
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (float(c) for c in quat_wxyz)


def _stamp(t):
    sec = math.floor(t)
    return Time(sec=int(sec), nanosec=int((t - sec) * 1e9))


class Bridge(Node):
    def __init__(self, a):
        super().__init__("vsss_mujoco")
        self.a = a
        self.scene = scene_from_args(a)
        self.m = self.scene.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.robots = [SimRobot(self, self.m, self.scene, r) for r in self.scene.robots]
        prefixes = [f"{r.name}/" for r in self.scene.robots]
        self.drive = (PlanarDrive(self.m, self.scene.p, prefixes, self.scene.q) if self.scene.model == "planar"
                      else DriveList(self.m, self.scene.p, prefixes))
        ball = self.m.joint("ball")
        self.ball_q, self.ball_v = self.m.jnt_qposadr[ball.id], self.m.jnt_dofadr[ball.id]
        self.reset_requested = False

        self.tf = TransformBroadcaster(self)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_ball = self.create_publisher(Odometry, "/ball/odom", 10)
        self.pub_img = self.create_publisher(Image, "/overhead_camera/image_raw", 2)
        self.pub_info = self.create_publisher(CameraInfo, "/overhead_camera/camera_info", 2)
        self.create_service(Empty, "/reset", self._on_reset)

        cam = self.scene.cam
        self.renderer = mujoco.Renderer(self.m, cam["height"], cam["width"])
        self.cam_info = None if cam["ortho"] else self._camera_info(cam)
        st = TransformStamped()  # ROS optical frame: x right (= field +x), y down (= field -y), z forward (down)
        st.header.frame_id, st.child_frame_id = "field", "overhead_camera_optical"
        st.transform.translation.z = float(cam["z"])
        st.transform.rotation.x, st.transform.rotation.w = 1.0, 0.0
        self.static_tf = StaticTransformBroadcaster(self)
        self.wall0 = time.time()
        st.header.stamp = _stamp(self.wall0)
        self.static_tf.sendTransform(st)
        if cam["ortho"]:
            self.get_logger().info(f"orthographic camera: {cam['fovy'] / cam['height'] * 1e3:.2f} mm per pixel, "
                                   f"image centre = field centre")
        self._reset()

    @staticmethod
    def _camera_info(cam):
        w, h = cam["width"], cam["height"]
        f = (h / 2) / math.tan(math.radians(cam["fovy"]) / 2)  # square pixels
        cx, cy = (w - 1) / 2, (h - 1) / 2
        info = CameraInfo(width=w, height=h, distortion_model="plumb_bob")
        info.header.frame_id = "overhead_camera_optical"
        info.d = [0.0] * 5
        info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _on_reset(self, request, response):  # rclpy thread: the sim loop does the reset
        self.reset_requested = True
        return response

    def _reset(self):
        mujoco.mj_resetData(self.m, self.d)  # qpos0 = the start poses the scene was built with
        mujoco.mj_forward(self.m, self.d)
        self.drive.reset(self.d)
        for r in self.robots:
            r.duty[:] = 0
            r.cmd = np.zeros(3)

    def publish_state(self):
        d = self.d
        stamp = _stamp(self.wall0 + d.time)
        self.pub_clock.publish(Clock(clock=stamp))
        tfs = []
        angle, speed, torque = self.drive.wheel_state(d)
        for i, r in enumerate(self.robots):
            r.publish(self.m, d, stamp, tfs, (angle[i], speed[i], torque[i]))
        self.tf.sendTransform(tfs)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, "field", "ball"
        _set_pose(od.pose.pose, d.qpos[self.ball_q:self.ball_q + 3], (1.0, 0.0, 0.0, 0.0))
        _set_vec(od.twist.twist.linear, d.qvel[self.ball_v:self.ball_v + 3])
        self.pub_ball.publish(od)

    def publish_camera(self):
        stamp = _stamp(self.wall0 + self.d.time)
        self.renderer.update_scene(self.d, camera="overhead")
        img = self.renderer.render()
        msg = Image(height=img.shape[0], width=img.shape[1], encoding="rgb8", is_bigendian=0, step=img.shape[1] * 3)
        msg.header.stamp, msg.header.frame_id = stamp, "overhead_camera_optical"
        msg.data = array.array("B", img.tobytes())
        self.pub_img.publish(msg)
        if self.cam_info is not None:
            self.cam_info.header.stamp = stamp
            self.pub_info.publish(self.cam_info)

    def run(self, viewer=None):
        a, m, d = self.a, self.m, self.d
        ctrl_dt = 1.0 / a.rate_hz
        n_sub = max(1, round(ctrl_dt / m.opt.timestep))
        ctrl_dt = n_sub * m.opt.timestep
        cam_dt = 1.0 / a.cam_hz
        next_cam = 0.0
        next_view = 0.0
        wall_start, sim_start = time.monotonic(), d.time
        log_wall, log_sim = wall_start, d.time
        while rclpy.ok() and (viewer is None or viewer.is_running()):
            if self.reset_requested:
                self.reset_requested = False
                self._reset()
                next_cam = 0.0
                wall_start, sim_start = time.monotonic(), d.time
                self.wall0 = time.time()
            for r in self.robots:
                r.update_duty(ctrl_dt, a)
            duty = np.array([r.duty for r in self.robots])
            for _ in range(n_sub):
                self.drive.step(d, duty)
            self.publish_state()
            if d.time >= next_cam:
                self.publish_camera()
                next_cam += cam_dt
            now = time.monotonic()
            if viewer is not None and now >= next_view:
                viewer.sync()
                next_view = now + 1 / 60
            if a.real_time > 0:  # pace sim time to wall time
                ahead = (d.time - sim_start) / a.real_time - (now - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.5:  # can't keep up: don't try to catch up in a burst later
                    wall_start, sim_start = now, d.time
            if now - log_wall > 10:
                self.get_logger().info(f"sim time {d.time:.0f} s, running at {(d.time - log_sim) / (now - log_wall):.2f}x real time")
                log_wall, log_sim = now, d.time


def _spin(node):
    try:
        rclpy.spin(node)
    except Exception:
        if rclpy.ok():
            raise  # a real error, not Ctrl+C shutting the context down


def main():
    ap = argparse.ArgumentParser(description="ROS 2 bridge for the VSSS MuJoCo scene (see the module docstring)")
    add_scene_args(ap)
    ap.add_argument("--cam_hz", type=float, default=60.0, help="overhead camera frame rate (sim time)")
    ap.add_argument("--rate_hz", type=float, default=100.0, help="control / state publishing rate (sim time)")
    ap.add_argument("--ramp", type=float, default=0.2, help="seconds for the duty to ramp 0 -> 100 %% (0 = no ramp)")
    ap.add_argument("--ff_gain", type=float, default=1.0, help="scale on the open-loop duty feed-forward")
    ap.add_argument("--cmd_timeout", type=float, default=0.0,
                    help="stop a robot after this many seconds without cmd_vel (0 = keep the last command)")
    ap.add_argument("--real_time", type=float, default=1.0, help="sim speed vs wall clock (0 = as fast as possible)")
    ap.add_argument("--no_viewer", action="store_true", help="don't open the MuJoCo 3D viewer")
    a = ap.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = Bridge(a)
    spin = threading.Thread(target=_spin, args=(node,), daemon=True)
    spin.start()
    names = ", ".join(r.name for r in node.robots)
    node.get_logger().info(f"robots: {names}; send geometry_msgs/Twist to /<robot>/cmd_vel. "
                           f"Camera: /overhead_camera/image_raw ({a.cam_width}x{a.cam_height} @ {a.cam_hz:g} Hz)")
    try:
        if a.no_viewer:
            node.run()
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(node.m, node.d) as viewer:
                node.run(viewer)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error, not Ctrl+C shutting the context down mid-publish
    finally:
        node.renderer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
