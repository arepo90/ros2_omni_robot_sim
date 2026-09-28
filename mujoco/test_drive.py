#!/usr/bin/env python3
"""Drive the generated omni robot open-loop through the wheel Jacobian and check
that the chassis actually does what was commanded.

  python test_drive.py --preset vsss   # the real robot: PWM + self-locking worm gears
  python test_drive.py                 # generic robot, servo motors (like the Gazebo sim)
  python test_drive.py --motor dc      # DC-motor model

With motor dc / worm it also runs a full-duty launch, a hard stop (duty 0 at top speed)
and a push test (how hard you must push the unpowered robot before it moves).
"""
import argparse
import time

import mujoco
import numpy as np

from drive import Drive
from omni_mjcf import add_param_args, build_mjcf, describe_rollers, params_from_args, roller_geometry, wheel_jacobian

G = 9.81


def yaw_of(q):
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def pitch_of(q):
    return np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))


def body_twist_measured(d):
    """(vx, vy, wz) of the chassis in its own frame (freejoint: lin vel world, ang vel local)."""
    c, s = np.cos(yaw_of(d.qpos[3:7])), np.sin(yaw_of(d.qpos[3:7]))
    v = d.qvel[:3]
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], d.qvel[5]])


def settle(m, d, t=0.3, drive=None):
    mujoco.mj_resetData(m, d)
    for _ in range(int(t / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if drive is not None:
        drive.reset(d)


def open_loop(p, w):
    """Wheel command for wheel speeds w: the speed itself (servo) or a duty cycle from the
    no-load motor curve (dc / worm), which is all a robot without encoders can do."""
    if p.motor == "servo":
        return w
    return w / (p.no_load_speed * (p.supply_voltage or p.nominal_voltage) / p.nominal_voltage)


def track(m, d, p, drive, cmd, T=1.5):
    settle(m, d, drive=drive)
    u = open_loop(p, wheel_jacobian(p) @ np.array(cmd))
    n = int(T / m.opt.timestep)
    meas = []
    for k in range(n):
        drive.step(d, u)
        if k > n // 2:
            meas.append(body_twist_measured(d))
    meas = np.array(meas)
    return meas.mean(0), meas.std(0)


def _full_ahead(p):
    dirn = wheel_jacobian(p) @ np.array([1.0, 0, 0])
    return dirn / np.abs(dirn).max()


def _accel(vx, dt, window=0.02):  # 20 ms averages: single-step derivatives are contact noise
    w = int(window / dt)
    return (vx[w:] - vx[:-w]) / (w * dt)


def launch_test(m, d, p, drive, T=0.8):
    """Full duty straight ahead (+x) from standstill."""
    J = wheel_jacobian(p)
    settle(m, d, drive=drive)
    u = _full_ahead(p)
    vx, slip, pitch, air = [], [], [], 0
    for _ in range(int(T / m.opt.timestep)):
        drive.step(d, u)
        vx.append(body_twist_measured(d)[0])
        slip.append(np.linalg.pinv(J)[0] @ d.sensordata[-p.n_wheels:] - vx[-1])  # wheel-odometry vx - true vx
        pitch.append(pitch_of(d.qpos[3:7]))
        air += d.ncon == 0
    vx = np.array(vx)
    t90 = np.argmax(vx > 0.9 * vx[-1]) * m.opt.timestep
    print(f"  launch: top speed {vx[-1]:.2f} m/s, 0->90% in {t90*1e3:.0f} ms, peak accel {_accel(vx, m.opt.timestep).max():.1f} m/s^2, "
          f"max pitch {np.abs(pitch).max():.1f} deg, airborne {air*m.opt.timestep*1e3:.0f} ms, "
          f"peak wheelspin {np.abs(slip).max():.2f} m/s")


def stop_test(m, d, p, drive, T_run=0.8, T_stop=1.2):
    """Full duty ahead, then duty to 0 (sudden, and ramped over 200 ms): stopping and pitching."""
    for ramp in (0.0, 0.2):
        settle(m, d, drive=drive)
        u = _full_ahead(p)
        for _ in range(int(T_run / m.opt.timestep)):
            drive.step(d, u)
        v0, x0 = body_twist_measured(d)[0], d.qpos[0]
        vx, pitch = [], []
        for k in range(int(T_stop / m.opt.timestep)):
            drive.step(d, u * (max(0.0, 1 - k * m.opt.timestep / ramp) if ramp else 0.0))
            vx.append(body_twist_measured(d)[0])
            pitch.append(pitch_of(d.qpos[3:7]))
        vx = np.array(vx)
        t_stop = np.argmax(np.abs(vx) < 0.02) * m.opt.timestep
        lift = 2 * p.wheel_R * np.cos(np.pi / 4) * np.sin(np.radians(np.abs(pitch).max()))
        how = "sudden stop" if not ramp else f"{ramp*1e3:.0f} ms ramp "
        print(f"  {how} from {v0:.2f} m/s: stopped in {t_stop*1e3:.0f} ms over {(d.qpos[0]-x0)*1e3:.0f} mm, "
              f"peak decel {-_accel(vx, m.opt.timestep).min():.1f} m/s^2, max pitch {np.abs(pitch).max():.1f} deg "
              f"(~{lift*1e3:.0f} mm lift)")


def push_test(m, d, p, drive, F_max=4.0, T=4.0, sliding=0.05):
    """Unpowered robot, horizontal push at axle height ramping 0 -> F_max: the force at which it
    slides (faster than `sliding` m/s; soft contacts creep a few mm/s before that)."""
    mass = m.body_subtreemass[1]
    out = []
    for name, ang, pred in [("x", 0.0, 0.5 ** 0.5), ("diagonal", 45.0, 0.5)]:
        settle(m, d, drive=drive)
        dirn = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
        F_break = None
        for k in range(int(T / m.opt.timestep)):
            F = F_max * k * m.opt.timestep / T
            d.qfrc_applied[:] = 0
            mujoco.mj_applyFT(m, d, F * dirn, np.zeros(3), d.xpos[1].copy(), 1, d.qfrc_applied)
            drive.step(d, np.zeros(p.n_wheels))
            if np.hypot(d.qvel[0], d.qvel[1]) > sliding:
                F_break = F
                break
        d.qfrc_applied[:] = 0
        tag = f"{F_break:.2f} N" if F_break else f"> {F_max:g} N"
        hint = f" (locked-wheel prediction {pred:.3f} mu m g = {pred*p.friction*mass*G:.2f} N)" if p.motor == "worm" else ""
        out.append(f"along {name}: {tag}{hint}")
    print("  push, unpowered: " + "; ".join(out))


def main():
    ap = argparse.ArgumentParser(description="--preset vsss for the real robot; any OmniParams field can be "
                                             "overridden, e.g. --rows 1 --rollers_per_row 10")
    add_param_args(ap)
    ap.add_argument("--save", help="also write the MJCF here")
    a = vars(ap.parse_args())
    save = a.pop("save")
    p = params_from_args(a)
    xml = build_mjcf(p)
    if save:
        open(save, "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    drive = Drive(m, p)
    mass = m.body_subtreemass[1]
    g = roller_geometry(p)
    print("wheel:", describe_rollers(p, g))
    r_kin = p.kinematic_radius or p.wheel_radius
    print(f"  kinematics use r = {r_kin*1e3:.2f} mm; geometric r_eff / r = {g.r_eff/r_kin*100:.1f}%")
    print(f"model: {m.nbody-1} bodies, {m.njnt} joints, {m.nu} motors ({p.motor}), mass {mass*1e3:.0f} g, "
          f"dt {m.opt.timestep*1e3:g} ms")
    settle(m, d, 0.5)
    print(f"  rest height of axle {d.qpos[2]*1e3:.2f} mm (ideal {p.wheel_radius*1e3:.2f}), "
          f"{d.ncon} contacts, roll/pitch ok: {abs(d.qpos[4])<1e-3 and abs(d.qpos[5])<1e-3}")

    cmds = [(0.5, 0, 0), (0, 0.5, 0), (0.35, 0.35, 0), (0, 0, 6.0), (0.3, 0, 3.0)]
    t0 = time.perf_counter(); sim_t = 0
    for c in cmds:
        mu, sd = track(m, d, p, drive, c)
        sim_t += 1.8
        ratio = [f"{mu[i]/c[i]*100:5.1f}%" if c[i] else "   -  " for i in range(3)]
        print(f"  cmd vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.1f} -> "
              f"vx={mu[0]:+.3f} vy={mu[1]:+.3f} wz={mu[2]:+.2f}  tracking {' '.join(ratio)}  ripple(std) {np.round(sd,3)}")
    print(f"  realtime factor ~{sim_t/(time.perf_counter()-t0):.0f}x")
    if p.motor in ("dc", "worm"):
        launch_test(m, d, p, drive)
        stop_test(m, d, p, drive)
        push_test(m, d, p, drive)


if __name__ == "__main__":
    main()
