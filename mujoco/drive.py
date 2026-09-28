"""Wheel drive models that need per-step logic, behind one interface.

    drive = Drive(m, p)
    drive.reset(d)
    drive.step(d, u)   # one physics step with wheel command u (one value per wheel)

u is a wheel speed [rad/s] for motor="servo" and a PWM duty cycle in [-1, 1] for
motor="dc" and motor="worm".

motor="worm" is a DC motor behind a worm gear, driven by plain PWM. The motor has its
own speed and angle, and meets the wheel through the gear's backlash:

  * inside the backlash the wheel turns freely;
  * on a tooth flank the gear is a stiff contact (the MJCF velocity actuator plus a
    position correction), so a held wheel keeps its angle instead of creeping;
  * the flank tells who is driving. If the motor pushes the wheel, the motor feels the full
    load, and the linear DC-motor curve limits the torque. If the wheel pushes the motor
    (braking, being pushed, coasting), a self-locking worm won't turn. The gear then holds
    the wheel at the motor's pace (up to worm_lock_torque), and the motor feels only
    backdrive_efficiency of that load.

Cutting power therefore stops the wheels almost at once, and a pushed robot skids
instead of rolling. Gearbox friction (wheel_frictionloss) gives the PWM deadband.
"""
import mujoco
import numpy as np

_EPS = 1e-3  # [rad/s] below this the motor counts as stopped
_KP = 200.0  # [1/s] flank position correction; with coupling_kv this is the gear stiffness


class Drive:
    def __init__(self, m: mujoco.MjModel, p):
        self.m, self.p = m, p
        self.volts = p.supply_voltage or p.nominal_voltage
        jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_{i}") for i in range(1, p.n_wheels + 1)]
        self.dof = np.array([m.jnt_dofadr[j] for j in jnt])
        self.qadr = np.array([m.jnt_qposadr[j] for j in jnt])
        if p.pwm_decay not in ("brake", "coast") or p.stop_mode not in ("brake", "coast"):
            raise ValueError("pwm_decay and stop_mode must be 'brake' or 'coast'")
        if p.motor == "worm" and p.coupling_kv * m.opt.timestep > p.wheel_armature:
            raise ValueError(f"coupling_kv must be <= wheel_armature / timestep = {p.wheel_armature/m.opt.timestep:.3g} "
                             "for the explicit motor-gear coupling to be stable")
        self.reset()

    def reset(self, d: mujoco.MjData = None):
        n = self.p.n_wheels
        self.w_motor = np.zeros(n)  # worm: motor speed and angle, in wheel units
        self.theta_motor = d.qpos[self.qadr].copy() if d is not None else np.zeros(n)

    def step(self, d: mujoco.MjData, u):
        p = self.p
        if p.motor == "servo":
            d.ctrl[:] = u
            mujoco.mj_step(self.m, d)
            return
        duty = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
        if p.motor == "dc":
            d.ctrl[:] = duty * self.volts
            mujoco.mj_step(self.m, d)
            return
        half = p.gear_backlash / 2
        e = self.theta_motor - d.qpos[self.qadr]  # > 0: motor ahead of the wheel
        over = np.where(e > half, e - half, np.where(e < -half, e + half, 0.0))
        w_wheel = d.qvel[self.dof]
        target = self.w_motor + _KP * over
        # a flank can push, never pull: free wheel (no torque) inside the backlash or when separating
        target = np.where(over > 0, np.maximum(target, w_wheel), np.where(over < 0, np.minimum(target, w_wheel), w_wheel))
        d.ctrl[:] = target
        mujoco.mj_step(self.m, d)
        self._update_motor(d, duty, over)

    def _update_motor(self, d, duty, over):
        p, dt = self.p, self.m.opt.timestep
        J, tf = p.wheel_armature, p.wheel_frictionloss
        tau_v = p.stall_torque / p.nominal_voltage * duty * self.volts  # stall torque at this duty
        b = p.stall_torque / p.no_load_speed  # back-EMF damping
        F = d.actuator_force[: p.n_wheels]  # torque the gear put on each wheel this step
        w = self.w_motor
        moving = np.abs(w) > _EPS
        flank = np.sign(over)
        driving = (flank != 0) & np.where(moving, flank == np.sign(w), flank == np.sign(tau_v))
        load = np.where(driving, F, p.backdrive_efficiency * F)  # the force MuJoCo actually applied
        # a disconnected motor produces no torque at all, not even back-EMF braking
        off = np.zeros(duty.shape, dtype=bool)
        if p.pwm_decay == "coast":  # motor disconnected in PWM off-time: it can push, never brake
            off |= (duty == 0) | (np.sign(tau_v - b * w) != np.sign(duty))
        if p.stop_mode == "coast":  # duty 0 switches the driver outputs off
            off |= duty == 0
        tau_v, b = np.where(off, 0.0, tau_v), np.where(off, 0.0, b)
        net = tau_v - b * w - load
        fdir = np.where(moving, np.sign(w), np.sign(net))
        # back-EMF implicit, gear load explicit (stable for coupling_kv * dt <= wheel_armature)
        new = (J * w + dt * (tau_v - load - tf * fdir)) / (J + dt * b)
        stuck = (np.abs(net) <= tf) & (~moving | (np.sign(new) != np.sign(w)))  # stiction
        self.w_motor = np.where(stuck, 0.0, new)
        self.theta_motor = self.theta_motor + self.w_motor * dt
