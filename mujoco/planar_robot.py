#!/usr/bin/env python3
"""Fast planar model of the omni robot: a box sliding on the floor (x, y, yaw) pushed by one force
per wheel, with the same DC motor and self-locking rule as the roller model (drive.py).

  python3 planar_robot.py            # runs the same tests on this model and the roller model, side by side

What's kept from the roller model (omni_mjcf.py + drive.py, the reference):
  * wheel layout, mass and yaw inertia (computed from the roller model), the motor curve, gearbox
    friction, PWM decay and stop modes;
  * traction: each wheel pushes along its rolling direction with force k*(r*w - v_t), capped at
    mu*N, while its rollers let it slide freely along the axle (`roller_drag`, 0 by default);
  * self-locking: the motor feels the ground only when it drives the wheel. When the ground would
    back-drive it (braking, pushing, coasting), the wheel stays at the motor's speed and the motor
    feels nothing (backdrive_efficiency).
What's dropped: rollers, contacts under the wheels, pitch/roll and load transfer (no tipping or
lift on hard stops), the 12-contact vibration. ~30x real time per robot, ~20x for a full field
(roller model: ~6x and ~1x).

Self-locking is not simulated, it's one guard (see PlanarDrive): the ground may resist a motor but
never drive it. The motor speed is solved per step with backward Euler, so it's stable at any
step size (results agree to <1 mm between 0.5 and 4 ms steps), and a robot at duty 0 cannot move.

The wheel forces are MuJoCo actuators on sites at the wheel contact points (gain k, velocity
bias -k, force range +-mu*N). They are integrated explicitly (semi-implicit Euler): the implicit
integrators keep the velocity derivative of a capped force, which acts like extra mass while the
wheels slip (6.1 instead of 6.9 m/s^2 at the traction limit, 2 ms). Stability needs
k*dt < 2*(effective mass); at 2 ms the stiffest mode (yaw) is at ~0.8.

Losses the planar model can't produce by itself (rolling over the 12 rims, silicone) are fitted to
the roller model's steady-state tracking: `rolling_resistance` and `rolling_damping`, counted only
while the motor drives the wheel. `roller_drag` and `yaw_scrub` exist but weren't needed. The
roller model's launch and ramp-stop numbers are not fitted: they come from contact artefacts
(reference.md, section 6).
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from omni_mjcf import OmniParams, build_mjcf, make_params, roller_geometry, wheel_angles

_EPS = 1e-3  # [rad/s] below this the motor counts as stopped (as in drive.py)
G = 9.81
# Render groups. The MuJoCo viewer shows groups 0-2: a see-through case and the wheels (VIEW_GROUP).
# A camera that should see the robot as it is renders groups 0, 1 and CAMERA_GROUP instead: the
# opaque case (the collision box). vsss_field.camera_option() sets that up.
VIEW_GROUP, CAMERA_GROUP = 2, 3
SHELL_RGBA = (0.6, 0.6, 0.6, 0.2)  # see-through grey case, viewer only
HUB_RGBA, RIM_RGBA = (0.92, 0.92, 0.92, 1), (0.95, 0.45, 0.1, 1)  # display colours of the wheels


@dataclass
class PlanarParams:
    timestep: float = 0.002
    slip_speed: float = 0.03  # [m/s] wheel slip at which traction reaches mu*N (linear below)
    rolling_resistance: float = 0.03  # Coulomb rolling loss of a wheel: torque c*N*r, felt by the motor
    rolling_damping: float = 6e-5  # [N m s/rad] speed-proportional rolling loss per wheel, felt by the motor
    roller_drag: float = 0.0  # [N s/m] per wheel, rollers rolling along the axle
    yaw_scrub: float = 0.0  # [N m s/rad] yaw damping from roller scrub


def mass_properties(p: OmniParams):
    """Total mass and yaw inertia about the chassis centre, from the roller model."""
    m = mujoco.MjModel.from_xml_string(build_mjcf(p))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    def diag(i):  # entry of the joint-space inertia matrix
        e, out = np.zeros(m.nv), np.zeros(m.nv)
        e[i] = 1.0
        mujoco.mj_mulM(m, d, out, e)
        return out[i]
    return diag(0), diag(5)  # freejoint: total mass, yaw inertia about the chassis centre


def robot_spec(p: OmniParams, q: PlanarParams, rgba=(0.05, 0.05, 0.05, 1), display_wheels=True) -> mujoco.MjSpec:
    """One planar robot as an MjSpec whose root body is "chassis" (attach it with a name prefix).
    display_wheels adds turning wheels to look at in the viewer (~50 % more time per step)."""
    mass, izz = mass_properties(p)
    N = mass * G / p.n_wheels
    k = p.friction * N / q.slip_speed
    spec = mujoco.MjSpec()
    spec.option.timestep = q.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    half, z_bot, z_top = p.body_size / 2, p.ground_clearance, p.body_height
    body = spec.worldbody.add_body(name="chassis", pos=[0, 0, 0])
    body.add_joint(name="x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    body.add_joint(name="y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    body.add_joint(name="yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=q.yaw_scrub)
    body.explicitinertial = True
    body.mass = mass
    body.ipos = [0, 0, p.com_height]
    body.inertia = [izz, izz, izz]  # only yaw matters in the plane
    box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, (z_top - z_bot) / 2],
               pos=[0, 0, (z_top + z_bot) / 2], mass=0)
    body.add_geom(name="chassis", rgba=rgba, contype=4, conaffinity=5, group=CAMERA_GROUP, **box)
    body.add_geom(name="shell", rgba=SHELL_RGBA, contype=0, conaffinity=0, group=VIEW_GROUP, **box)
    body.add_site(name="imu", pos=[0, 0, p.wheel_radius], group=4)
    for i, th in enumerate(wheel_angles(p), start=1):
        c, s = np.cos(th), np.sin(th)
        body.add_site(name=f"contact_{i}", pos=[p.wheel_R * c, p.wheel_R * s, 0], size=[0.002, 0, 0], group=4)
        if display_wheels:
            _display_wheel(body, p, i, [p.wheel_R * c, p.wheel_R * s, p.wheel_radius], [-c, -s, 0])
        # traction along the rolling direction t = (-sin, cos): force = k * (ctrl - v_t), ctrl = r * w
        spec.add_actuator(name=f"traction_{i}", trntype=mujoco.mjtTrn.mjTRN_SITE, target=f"contact_{i}",
                          gear=[-s, c, 0, 0, 0, 0], gainprm=[k] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, 0, -k] + [0] * 7,
                          forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-p.friction * N, p.friction * N])
        # rollers: drag along the axle a = (cos, sin)
        spec.add_actuator(name=f"roller_{i}", trntype=mujoco.mjtTrn.mjTRN_SITE, target=f"contact_{i}",
                          gear=[c, s, 0, 0, 0, 0], gainprm=[0] * 10,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, 0, -q.roller_drag] + [0] * 7)
    spec.add_sensor(name="imu_quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    return spec


def _quat_z_to(v):
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, np.asarray(v, dtype=float))
    return q


def _display_wheel(chassis, p: OmniParams, i, pos, axle):
    """A wheel to look at, not to simulate: the hub and roller rims of the roller model on a hinge
    that PlanarDrive turns to the wheel's angle. No collisions, negligible mass."""
    wheel = chassis.add_body(name=f"wheel_{i}", pos=pos, quat=_quat_z_to(axle))  # z = axle, inward
    wheel.explicitinertial = True
    wheel.mass, wheel.inertia = 1e-5, [1e-9, 1e-9, 1e-9]
    wheel.add_joint(name=f"wheel_{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], armature=1e-6)
    vis = dict(contype=0, conaffinity=0, group=VIEW_GROUP, mass=0)
    g = roller_geometry(p)
    hub_r = p.hub_radius or 0.9 * g.d
    wheel.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hub_r, p.hub_thickness / 2, 0], rgba=HUB_RGBA, **vis)
    rows = [0.0] if p.rows == 1 else [-p.row_spacing / 2, p.row_spacing / 2]
    lobes = (-g.lobe_offset, g.lobe_offset) if p.roller_shape == "peanut" else (0.0,)
    for k, z in enumerate(rows):
        for j in range(p.rollers_per_row):
            phi = 2 * np.pi * j / p.rollers_per_row + k * np.pi / p.rollers_per_row
            centre = np.array([g.d * np.cos(phi), g.d * np.sin(phi), z])
            axis = np.array([-np.sin(phi), np.cos(phi), 0.0])  # roller axle, tangent to the wheel
            for off in lobes:
                wheel.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=centre + off * axis,
                               quat=_quat_z_to(axis), size=[p.roller_radius, p.roller_radius, g.half_len],
                               rgba=RIM_RGBA, **vis)


class PlanarDrive:
    """Motors and wheels of any number of planar robots, vectorised.

        drive = PlanarDrive(m, p, ["blue_0/", "blue_1/"], q)
        drive.step(d, duty)   # one physics step; duty: (n_robots, n_wheels) PWM duty in [-1, 1]

    Each wheel is geared rigidly to its motor. Every step the motor speed w' is solved with
    backward Euler (stable at any step size):

        J (w' - w) / dt = tau_motor(duty, w') - friction - load(w')

    load is the torque the wheel needs to turn at w' while the ground under it moves at v_t:
    r*F plus rolling losses, with F = traction = k*(r*w' - v_t) capped at mu*N. Self-locking is a
    single guard on that load: the ground may resist the motor but never drive it, so a load that
    would push the motor along counts as 0. The wheel then keeps the motor's own speed and the
    traction brakes the robot instead. MuJoCo applies the same F to the robot (actuator
    "traction_i", evaluated at the same v_t), so both sides see one force.
    """

    def __init__(self, m: mujoco.MjModel, p: OmniParams, prefixes, q: PlanarParams = None):
        q = q or PlanarParams()
        if p.backdrive_efficiency != 0:
            raise ValueError("the planar model is for self-locking gearboxes (backdrive_efficiency 0)")
        if p.pwm_decay not in ("brake", "coast") or p.stop_mode not in ("brake", "coast"):
            raise ValueError("pwm_decay and stop_mode must be 'brake' or 'coast'")
        self.m, self.p, self.q = m, p, q
        n = p.n_wheels
        self.shape = (len(prefixes), n)
        self.act = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{pre}traction_{i}")
                              for i in range(1, n + 1)] for pre in prefixes])
        if (self.act < 0).any():
            raise ValueError("model has no planar robots with these prefixes")
        jnt = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{pre}wheel_{i}")
                         for i in range(1, n + 1)] for pre in prefixes])
        self.wheel_q = m.jnt_qposadr[jnt] if (jnt >= 0).all() else None  # display wheels, if any
        self.wheel_v = m.jnt_dofadr[jnt] if (jnt >= 0).all() else None
        self.r = roller_geometry(p).r_eff  # the wheel rolls on its effective radius
        mass, _ = mass_properties(p)
        self.N = mass * G / n
        self.k = p.friction * self.N / q.slip_speed
        self.volts = p.supply_voltage or p.nominal_voltage
        self.reset()

    def reset(self, d: mujoco.MjData = None):
        self.w = np.zeros(self.shape)  # wheel (= geared motor) speed [rad/s]
        self.theta = np.zeros(self.shape)  # wheel angle, for display
        self.duty = np.zeros(self.shape)

    def step(self, d: mujoco.MjData, duty):
        self.duty = np.clip(np.asarray(duty, dtype=float).reshape(self.shape), -1.0, 1.0)
        mujoco.mj_step1(self.m, d)  # positions and velocities -> actuator_velocity = v_t now
        self.w = self._motor(d.actuator_velocity[self.act])
        self.theta = self.theta + self.w * self.m.opt.timestep
        d.ctrl[self.act] = self.r * self.w
        mujoco.mj_step2(self.m, d)  # traction F = clamp(k*(r*w - v_t)), integrate
        if self.wheel_q is not None:  # turn the display wheels
            d.qpos[self.wheel_q], d.qvel[self.wheel_v] = self.theta, self.w

    def _motor(self, v):
        p, q, r, k = self.p, self.q, self.r, self.k
        a = p.wheel_armature / self.m.opt.timestep
        w, duty, Fmax = self.w, self.duty, p.friction * self.N
        tau = p.stall_torque / p.nominal_voltage * duty * self.volts  # torque at stall for this duty
        b = np.full(self.shape, p.stall_torque / p.no_load_speed)  # back-EMF
        off = np.zeros(self.shape, dtype=bool)  # driver outputs disconnected: no torque, no back-EMF
        if p.pwm_decay == "coast":
            off |= (duty == 0) | (np.sign(tau - b * w) != np.sign(duty))
        if p.stop_mode == "coast":
            off |= duty == 0
        tau, b = np.where(off, 0.0, tau), np.where(off, 0.0, b)
        s = np.where(np.abs(w) > _EPS, np.sign(w), np.sign(tau))  # direction the motor turns (or starts)
        fric = (p.wheel_frictionloss + q.rolling_resistance * self.N * r) * s
        c1 = q.rolling_damping
        # 1) unloaded: the ground doesn't resist (or would push, which the worm blocks)
        w0 = (a * w + tau - p.wheel_frictionloss * s) / (a + b)
        load0 = r * np.clip(k * (r * w0 - v), -Fmax, Fmax) + q.rolling_resistance * self.N * r * s + c1 * w0
        loaded = load0 * s > 0
        # 2) loaded, wheel gripping: load = r*k*(r*w' - v) + rolling losses
        w1 = (a * w + tau - fric + r * k * v) / (a + b + c1 + k * r * r)
        # 3) loaded, wheel slipping: traction at its cap
        w2 = (a * w + tau - fric - r * Fmax * s) / (a + b + c1)
        new = np.where(loaded, np.where(np.abs(k * (r * w1 - v)) > Fmax, w2, w1), w0)
        return np.where(new * s > 0, new, 0.0)  # friction stops it, never reverses it

    def wheel_state(self, d: mujoco.MjData):
        """(angle, speed, torque on the wheel) per robot and wheel."""
        return self.theta, self.w, self.r * d.actuator_force[self.act]


# --- side-by-side tests: planar vs roller model ----------------------------------------------

class _Rig:
    """One robot on a floor, either model, with a common read-out."""

    def __init__(self, model, p: OmniParams, q: PlanarParams):
        from drive import Drive
        if model == "roller":
            self.m = mujoco.MjModel.from_xml_string(build_mjcf(p))
            self.d = mujoco.MjData(self.m)
            self.drive = Drive(self.m, p)
        else:
            spec = mujoco.MjSpec()
            spec.option.timestep, spec.option.integrator = q.timestep, mujoco.mjtIntegrator.mjINT_EULER
            spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
            spec.worldbody.add_frame().attach_body(robot_spec(p, q).body("chassis"), "r/", "")
            self.m = spec.compile()
            self.d = mujoco.MjData(self.m)
            self.drive = PlanarDrive(self.m, p, ["r/"], q)
        self.step = lambda u: self.drive.step(self.d, u)
        self.body = self.m.body("chassis" if model == "roller" else "r/chassis").id
        self.p = p

    def reset(self, settle=0.3):
        mujoco.mj_resetData(self.m, self.d)
        self.drive.reset(self.d)
        for _ in range(int(settle / self.m.opt.timestep)):
            self.step(np.zeros(self.p.n_wheels))

    def twist(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.body, v, 1)
        return np.array([v[3], v[4], v[2]])  # body-frame vx, vy, wz

    def xy(self):
        return self.d.xpos[self.body][:2].copy()

    def yaw(self):
        q = self.d.xquat[self.body]
        return np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def compare(p: OmniParams = None, q: PlanarParams = None):
    import time
    from omni_mjcf import wheel_jacobian
    p = p or make_params("vsss")
    q = q or PlanarParams()
    J = wheel_jacobian(p)
    ff = p.no_load_speed * (p.supply_voltage or p.nominal_voltage) / p.nominal_voltage
    full = J @ [1, 0, 0] / np.abs(J @ [1, 0, 0]).max()
    rows = {}
    for model in ("roller", "planar"):
        rig, out = _Rig(model, p, q), []
        dt = rig.m.opt.timestep
        t0, sim = time.perf_counter(), 0.0
        for cmd in [(0.5, 0, 0), (0.3, 0, 0), (0.35, 0.35, 0), (0, 0, 6.0), (0.3, 0, 3.0)]:
            rig.reset()
            u = J @ np.array(cmd) / ff
            meas = []
            for s in range(int(1.5 / dt)):
                rig.step(u)
                if s > 0.75 / dt:
                    meas.append(rig.twist())
            mu = np.mean(meas, 0)
            out.append(" ".join(f"{mu[i] / cmd[i] * 100:.0f}%" for i in range(3) if cmd[i]))
            sim += 1.8
        rtf = sim / (time.perf_counter() - t0)
        rig.reset()
        vx = []
        for _ in range(int(0.8 / dt)):
            rig.step(full)
            vx.append(rig.twist()[0])
        vx = np.array(vx)
        out.append(f"{vx[-1]:.2f} m/s, {np.argmax(vx > 0.9 * vx[-1]) * dt * 1e3:.0f} ms")
        for stop in ("brake", "coast", "ramp"):
            rig.drive.p = make_params("vsss", stop_mode="coast") if stop == "coast" else p
            rig.reset()
            for _ in range(int(0.8 / dt)):
                rig.step(full)
            x0 = rig.xy()
            for s in range(int(1.2 / dt)):
                rig.step(full * (max(0.0, 1 - s * dt / 0.2) if stop == "ramp" else 0.0))
            out.append(f"{np.linalg.norm(rig.xy() - x0) * 1e3:.0f} mm")
            rig.drive.p = p
        push = []
        for ang in (0.0, 45.0):
            rig.reset()
            dirn = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
            Fb = None
            for s in range(int(4.0 / dt)):
                Fm = 4.0 * s * dt / 4.0
                rig.d.xfrc_applied[rig.body, :3] = Fm * dirn
                rig.step(np.zeros(p.n_wheels))
                if np.linalg.norm(rig.twist()[:2]) > 0.05:
                    Fb = Fm
                    break
            rig.d.xfrc_applied[:] = 0
            push.append(f"{Fb:.2f}" if Fb else ">4")
        out.append(" / ".join(push) + " N")
        rig.reset()
        for _ in range(int(10 / dt)):
            rig.step(np.zeros(p.n_wheels))
        out.append(f"{rig.yaw():+.2f} deg, {np.linalg.norm(rig.twist()[:2]) * 1e3:.1f} mm/s")
        out.append(f"{rtf:.0f}x")
        rows[model] = out
    names = ["vx 0.5", "vx 0.3", "diagonal 0.35/0.35", "spin 6 rad/s", "arc 0.3 + 3 rad/s", "full duty launch",
             "sudden stop (brake)", "sudden stop (coast)", "200 ms ramp stop", "push x / diagonal", "idle 10 s", "real time"]
    print(f"{'test':22s} {'roller':>22s} {'planar':>22s}")
    for i, nm in enumerate(names):
        print(f"{nm:22s} {rows['roller'][i]:>22s} {rows['planar'][i]:>22s}")


if __name__ == "__main__":
    compare()
