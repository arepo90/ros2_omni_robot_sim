# MuJoCo port of the omni-wheel robots

```bash
pip install mujoco numpy          # + xacro for import_repo_urdf.py
python omni_mjcf.py               # writes vsss_omni4.xml (placeholder 75 mm, 4-wheel VSSS robot)
python test_drive.py              # drives it through the wheel Jacobian and checks what it does
python -m mujoco.viewer --mjcf=vsss_omni4.xml
```

| file | what it is |
|---|---|
| `omni_mjcf.py` | Parametric MJCF generator. Any wheel count, layout, size, roller count, motor model. Also has `wheel_jacobian()` / `body_twist()` (same math as `src/kinematics.cpp`). |
| `test_drive.py` | Open-loop tracking test (vx, vy, diagonal, spin, arc) plus a full-throttle launch test for `--motor dc`. |
| `import_repo_urdf.py` | 1:1 import of the Gazebo robots (`3w_v2`, `4w`, `5w`, `6w`) for comparison. |

## How the wheel model works

It uses the same structure as the Gazebo URDFs:

```
chassis (free body)
 └─ wheel_i   hinge, driven. Axis points inward along the radius, at angle θ_i = offset + 360°·i/N
     └─ roller_i_kj   hinge, passive. Axis is tangent to the wheel. rows × rollers_per_row per wheel
```

The rollers are ellipsoids whose length is tuned so the wheel's rolling radius stays within
about 0.1 mm of `wheel_radius`. Ellipsoid–plane contact is analytic and smooth. A faceted
mesh roller (`--roller_shape mesh`) caused visible fake vibration at speed.

Kinematics, for wheel i at angle θ_i, distance R (centre to wheel mid-plane) and rolling radius r:

```
ω_i = ( -sin θ_i · vx + cos θ_i · vy + R · ωz ) / r        # inverse kinematics
[vx vy ωz] = pinv(J) · ω                                   # odometry (least squares, N > 3)
```

With 4 wheels the system is over-determined. One wheel-speed pattern (+,−,+,− in X layout)
makes the wheels fight each other and moves nothing, so r and R must match the real robot
or the wheels scrub.

## Parameters to replace with your robot's

The defaults are placeholders. Measure or look up the following:

- `wheel_radius`, `wheel_R`, `heading_offset_deg` (−45 = X layout, 0 = + layout)
- `rows`, `rollers_per_row`, `roller_radius`, `row_spacing` (from the wheel you buy or print)
- `body_mass`, `com_height`, `hub_mass`, `roller_mass`
- motor, measured at the wheel after the gearbox: `stall_torque`, `no_load_speed`,
  `nominal_voltage`, `wheel_armature` (rotor inertia × gear²)
- `friction` (roller material on the field surface), `contact_timeconst`

`--motor servo` is an ideal speed loop with a torque cap. It is closest to what the Gazebo sim
does, which is ideal velocity joints with no motor model. `--motor dc` is a voltage-driven
linear DC motor, τ = (τ_stall/V)·u − (τ_stall/ω₀)·ω, which exposes torque and traction limits.

## Validation (defaults)

| test | servo | dc (open-loop voltage) |
|---|---|---|
| vx / vy / diagonal tracking | 100 % | 99.6–99.9 % |
| spin 6 rad/s | 99.9 % | 99.6 % |
| arc (0.3 m/s + 3 rad/s) | 100 % | 99–100 % |
| speed | ~15× real time | ~15× real time |

A full-throttle launch with the placeholder DC motors is traction-limited. The motors can push
about 11× more force than friction allows (μ·m·g ≈ 1.5 N). The wheels spin, encoder odometry is off by more than 1 m/s,
and the robot chatters. Limit acceleration to about μ·g in firmware.

## Notes

- Robot geoms only collide with the floor (and other robots' chassis). Gazebo's `selfCollide`
  on rollers is not reproduced.
- The repo's collision meshes can't be reused as-is. MuJoCo convex-hulls meshes, and the
  chassis hull swallows the wheels. `import_repo_urdf.py` shows this and works around it.
- `src/kinematics.cpp` hard-codes `ROBOT_RADIUS 0.088` (from the original 3w model). The
  3w_v2/4w/5w/6w wheels sit at 0.1028 m, so those robots turn at about 87 % of the commanded
  ωz. `import_repo_urdf.py` reproduces this.
