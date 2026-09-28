#!/usr/bin/env python3
"""Drive the generated omni robot open-loop through the wheel Jacobian and check
that the chassis actually does what was commanded.

  python test_drive.py                 # servo motors (like the Gazebo sim)
  python test_drive.py --motor dc      # DC-motor model: also runs a full-throttle launch test
"""
import argparse
import time
from dataclasses import fields

import mujoco
import numpy as np

from omni_mjcf import OmniParams, add_param_args, build_mjcf, describe_rollers, params_from_args, roller_geometry, wheel_jacobian


def yaw_of(q):
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def body_twist_measured(d):
    """(vx, vy, wz) of the chassis in its own frame (freejoint: lin vel world, ang vel local)."""
    c, s = np.cos(yaw_of(d.qpos[3:7])), np.sin(yaw_of(d.qpos[3:7]))
    v = d.qvel[:3]
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], d.qvel[5]])


def settle(m, d, t=0.3):
    mujoco.mj_resetData(m, d)
    for _ in range(int(t / m.opt.timestep)):
        mujoco.mj_step(m, d)


def track(m, d, p, cmd, T=1.5):
    J = wheel_jacobian(p)
    settle(m, d)
    w = J @ np.array(cmd)
    if p.motor == "dc":  # open-loop voltage from the linear motor curve (no load)
        v_max = p.supply_voltage or p.nominal_voltage
        d.ctrl[:] = np.clip(w / p.no_load_speed * p.nominal_voltage, -v_max, v_max)
    else:
        d.ctrl[:] = w
    n = int(T / m.opt.timestep)
    meas = []
    for k in range(n):
        mujoco.mj_step(m, d)
        if k > n // 2:
            meas.append(body_twist_measured(d))
    meas = np.array(meas)
    return meas.mean(0), meas.std(0)


def launch_test(m, d, p, T=0.6):
    """Full voltage straight ahead (+x): acceleration, top speed, wheel slip."""
    J = wheel_jacobian(p)
    settle(m, d)
    dirn = J @ np.array([1.0, 0, 0])
    d.ctrl[:] = (p.supply_voltage or p.nominal_voltage) * dirn / np.abs(dirn).max()
    t, vx, slip, air = [], [], [], 0
    for k in range(int(T / m.opt.timestep)):
        mujoco.mj_step(m, d)
        wheel_w = d.sensordata[-p.n_wheels:]
        t.append(d.time)
        vx.append(body_twist_measured(d)[0])
        slip.append(np.linalg.pinv(J)[0] @ wheel_w - vx[-1])  # encoder-odometry vx minus true vx
        air += d.ncon == 0
    t, vx, slip = map(np.array, (t, vx, slip))
    v_top = vx[-1]
    t90 = t[np.argmax(vx > 0.9 * v_top)] - t[0]
    win = int(0.02 / m.opt.timestep)  # 20 ms window: single-step derivatives are contact noise
    acc = (vx[win:] - vx[:-win]) / (t[win:] - t[:-win])
    print(f"  launch: top speed {v_top:.2f} m/s, 0->90% in {t90*1e3:.0f} ms, "
          f"peak accel (20 ms avg) {acc.max():.1f} m/s^2 vs friction limit mu*g = {p.friction*9.81:.1f}, "
          f"airborne {air*m.opt.timestep*1e3:.0f} ms, peak odometry error from wheelspin {np.abs(slip).max():.2f} m/s")


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
    mass = m.body_subtreemass[1]
    g = roller_geometry(p)
    print("wheel:", describe_rollers(p, g))
    print(f"  expected translation tracking with nominal r: r_eff / r = {g.r_eff/p.wheel_radius*100:.1f}%")
    print(f"model: {m.nbody-1} bodies, {m.njnt} joints, {m.nu} motors, mass {mass*1e3:.0f} g, dt {m.opt.timestep*1e3:g} ms")
    settle(m, d, 0.5)
    print(f"  rest height of axle {d.qpos[2]*1e3:.2f} mm (ideal {p.wheel_radius*1e3:.2f}), "
          f"{d.ncon} contacts, roll/pitch ok: {abs(d.qpos[4])<1e-3 and abs(d.qpos[5])<1e-3}")

    cmds = [(0.5, 0, 0), (0, 0.5, 0), (0.35, 0.35, 0), (0, 0, 6.0), (0.3, 0, 3.0)]
    t0 = time.perf_counter(); sim_t = 0
    for c in cmds:
        mu, sd = track(m, d, p, c)
        sim_t += 1.8
        ratio = [f"{mu[i]/c[i]*100:5.1f}%" if c[i] else "   -  " for i in range(3)]
        print(f"  cmd vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.1f} -> "
              f"vx={mu[0]:+.3f} vy={mu[1]:+.3f} wz={mu[2]:+.2f}  tracking {' '.join(ratio)}  ripple(std) {np.round(sd,3)}")
    print(f"  realtime factor ~{sim_t/(time.perf_counter()-t0):.0f}x")
    if p.motor == "dc":
        launch_test(m, d, p)


if __name__ == "__main__":
    main()
