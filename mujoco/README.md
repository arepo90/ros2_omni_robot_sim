# MuJoCo port of the omni-wheel robots

```bash
pip install mujoco numpy          # + xacro for import_repo_urdf.py
python omni_mjcf.py --preset vsss # writes vsss_omni4.xml for the real robot (see below)
python test_drive.py --preset vsss   # drives it through the wheel Jacobian and checks what it does
python omni_mjcf.py               # without --preset: a generic placeholder 75 mm robot
python test_drive.py --rows 1 --rollers_per_row 10 --roller_radius 0.003   # any parameter can be overridden
python test_drive.py --roller_shape peanut --rows 1 --rollers_per_row 6 --roller_radius 0.0035
python -m mujoco.viewer --mjcf=vsss_omni4.xml
```

| file | what it is |
|---|---|
| `omni_mjcf.py` | Parametric MJCF generator. Any wheel count, layout, size, roller count, motor model. Also has `wheel_jacobian()` / `body_twist()` (same math as `src/kinematics.cpp`). |
| `drive.py` | `Drive(m, p).step(d, u)`: steps the sim with wheel commands (speeds for `servo`, PWM duty for `dc` / `worm`); holds the worm-gear model. With several robots: `Drive(m, p, prefix="blue_0/")`, `apply()` each, one `mj_step`, `update()` each. |
| `test_drive.py` | Open-loop tracking test (vx, vy, diagonal, spin, arc); for `dc` / `worm` also launch, sudden and ramped stop, and push tests. |
| `planar_robot.py` | Fast planar robot: a box on the floor (x, y, yaw) pushed by one traction force per wheel, same motor and self-locking rule. ~20× real time for a full field. `python3 planar_robot.py` compares it with the roller model. |
| `vsss_field.py` | The VSSS field with walls, goals, lines, the ball, up to 5 robots per team (planar or roller) and an overhead camera. |
| `ros_bridge.py` | ROS 2 (rclpy) bridge: `cmd_vel` per robot in; ground-truth odometry, IMU, wheels, duty, TF, `/clock` and the camera image out. |
| `import_repo_urdf.py` | 1:1 import of the Gazebo robots (`3w_v2`, `4w`, `5w`, `6w`) for comparison. |

## VSSS field and ROS 2 bridge

```bash
python3 vsss_field.py --blue 3 --yellow 3        # writes vsss_field.xml; python3 -m mujoco.viewer --mjcf=vsss_field.xml
source /opt/ros/humble/setup.bash
python3 ros_bridge.py                             # 3 blue robots, ball, overhead camera, MuJoCo viewer window
python3 ros_bridge.py --blue 3 --yellow 3 --no_viewer
python3 ros_bridge.py --model roller              # detailed roller-level robots (slow: 3v3 below real time)
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/blue_0/cmd_vel   # Shift+J/L strafe
ros2 run rqt_image_view rqt_image_view /overhead_camera/image_raw
ros2 topic echo /blue_0/odom        # or plot it: rqt_plot, plotjuggler
ros2 service call /reset std_srvs/srv/Empty
```

| topic | type | what |
|---|---|---|
| `/<robot>/cmd_vel` | `geometry_msgs/Twist` | in: body-frame `linear.x`, `linear.y` [m/s], `angular.z` [rad/s] |
| `/<robot>/odom` | `nav_msgs/Odometry` | ground truth: pose in `field`, twist in `<robot>/base_link` |
| `/<robot>/imu` | `sensor_msgs/Imu` | IMU at the chassis centre, noise-free |
| `/<robot>/wheels` | `sensor_msgs/JointState` | wheel angle, speed, torque on the wheel |
| `/<robot>/duty` | `std_msgs/Float64MultiArray` | PWM duty per wheel after the ramp |
| `/ball/odom` | `nav_msgs/Odometry` | ball position and velocity |
| `/overhead_camera/image_raw`, `/camera_info` | `sensor_msgs/Image` (rgb8), `CameraInfo` | 640×480 at 60 Hz by default |
| `/tf`, `/tf_static`, `/clock` | | `field` → `<robot>/base_link`, `field` → `overhead_camera_optical`; sim time |

Robots are `blue_0`..`blue_4` and `yellow_0`..`yellow_4`. The field frame has its origin at the
centre, x towards the yellow goal, z up. `cmd_vel` goes through what the firmware does: inverse
kinematics, the no-load duty feed-forward (`--ff_gain`), a ramp (`--ramp 0.2` s for 0 → 100 %),
and the motor model. It is open loop, so robots reach ~75–82 % of the commanded speed. A
command holds until the next one (`--cmd_timeout` to stop instead). Stamps and `/clock` are wall
start time + sim time; if the sim can't keep up (only with `--model roller`), set `use_sim_time`
in your tools.

**Robot model** (`--model`): `planar` (default) is a 7.5 cm box on the floor with one traction force
per wheel (`planar_robot.py`); `roller` is the roller-level model the rest of this README describes.
The planar model keeps the motor curve, rotor inertia, gearbox friction and the traction limit, and
drops rollers, pitch and vibration. Self-locking isn't simulated: it's one guard on the motor's load,
"the ground may resist the motor, never drive it". When the ground would push (braking, being pushed,
coasting) the wheel keeps the motor's own speed and the traction brakes the robot. The motor speed is
solved with backward Euler each step, so results don't depend on the step size (0.5–4 ms agree
to < 1 mm) and a robot at duty 0 can't move. Its losses are fitted to the roller model's steady-state
tracking (`python3 planar_robot.py`):

| test | roller | planar |
|---|---|---|
| open loop 0.5 / 0.3 m/s along x | 82 / 75 % | 82 / 75 % |
| diagonal, spin 6 rad/s, arc | 83, 74, 71/90 % | 84, 74, 73/93 % |
| full duty: top speed, 0 → 90 % | 0.76 m/s, 140 ms* | 0.76 m/s, 204 ms |
| sudden stop, brake / coast | 40 / 207 mm | 42 / 231 mm |
| 200 ms ramp stop | 67 mm* | 109 mm |
| push unpowered, x / diagonal | 1.72 / 1.39 N | 1.66 / 1.23 N |
| idle 10 s | still | still |
| speed, one robot | 6× real time | ~30× |

\* roller-model contact artefacts: on launch the robot gains more energy than the motors put in
(it passes the 6.9 m/s² traction limit), and in the ramp stop it stops, even rolls back, while its
wheels still turn forward. The planar values follow from the motor model (rotor inertia is still a
placeholder). On the full field the planar physics runs ~20× real time (~95 µs per 2 ms step) with
1 to 10 robots; the bridge holds real time for 3v3 with the viewer and camera, and runs ~3–4×
real time when unpaced (`--real_time 0`), mostly spent building ROS messages.

**Field:** FIRASim's Division B defaults (the league's simulator): 150 × 130 cm matte black floor,
5 cm walls 2.5 cm thick, 40 × 10 cm goals, 3 mm lines, 20 cm centre circle, 70 × 15 cm defense
areas, 7 cm corner triangles, orange golf ball (42.7 mm, 46 g). Check them against the current
rules. The ball's rolling resistance is a guess.

**Robot tops:** black 7.5 cm cube with the team's 3-colour pattern: team colour (blue / yellow)
across the front half, two ID colours side by side on the rear half (left, right seen from above
with the front up), 4 mm black borders. Colours are sampled from the team's pattern sheet: blue
(5, 11, 159), yellow (255, 230, 13), red (204, 0, 1), green (0, 204, 8), cyan (0, 170, 206), magenta
(205, 23, 220). Robot i uses pair i of the sheet's 10 (`vsss_field.ID_PAIRS`: RG, RC, GR, GC, GM, CR,
CG, CM, MG, MC). Lighting sums to 1 on surfaces facing the camera, so the tops render in exactly these
colours.

**Camera:** a pinhole camera looking straight down from `--cam_z 2.0` m, its field of view fitted to
the field (`--cam_fovy` to set it, `--cam_width/--cam_height`, `--cam_hz`). `camera_info` matches
the rendered image to 0.5 px. It sees perspective like a real overhead camera: a robot top 7 cm up
appears ~3.6 % further from the image centre than its footprint, 2.7 cm at the goal lines, so the
vision pipeline has to correct for marker height. `--cam_ortho` gives a flat map without parallax
(no `camera_info` then). Images are clean: no lens distortion, blur or noise.

Checks (roller model): each robot on the field tracks exactly like the lone robot in
`test_drive.py` (0.407 m/s for 0.5 m/s commanded along y, with 1, 3 or 6 robots), and
`test_drive.py` output is unchanged by the multi-robot `Drive`. Roller physics runs 4.7× real time
with 1 robot, 2.0× with 3, 1.08× with 6.

## The real robot (`--preset vsss`)

| spec | parameter |
|---|---|
| 7.5 cm base, wheels on the diagonals, centres 34 mm from the base centre | `heading_offset_deg -45`, `wheel_R 0.034` |
| wheel fits a Ø34.5 mm circle, hub a Ø31.4 mm circle | `wheel_radius 0.01725`, `hub_radius 0.0157` (the hub collides, to catch strikes) |
| 6 silicone (≤ Shore A40) spool rollers, 9.8 mm long, Ø6.7 mm ends, Ø5 mm waist | `roller_shape peanut`, `rows 1`, `rollers_per_row 6`, `roller_radius 0.00335`, `lobe_half_length 0.00128` |
| 230 g total, battery and motors at the bottom | `body_mass 0.214` (+ ~4 g per wheel, estimated), `com_height 0.025` |
| 4x GA12-N20 worm gearmotor, 12 V: 381 rpm, stall 235 gf·cm / 700 mA, no-load 30 mA | `motor worm`, `nominal_voltage 12`, `no_load_speed 39.9`, `stall_torque 0.0230`, `wheel_frictionloss 0.001` |
| self-locking worm gear (confirmed: a wheel won't turn by hand) | `backdrive_efficiency 0` |
| 3S LiPo | `supply_voltage 11.1` (12.6 full, ~10.5 empty) |
| 2× TB6612FNG, plain PWM, no encoders or current sensing | commands are duty cycles in [-1, 1] (`drive.Drive`); `pwm_decay brake`, `stop_mode brake` or `coast` |
| painted MDF field, "pretty grippy" | `friction 1.0` (estimate) |
| rolling radius to use in the kinematics | `kinematic_radius 0.0169` (estimate) |

Derived geometry: contact rims at ±15°, rolling radius 16.49–17.27 mm, **effective radius
17.06 mm**. The waist stays 0.85 mm and the hub 0.79 mm off the floor. A Hertz estimate for
A30–35 silicone gives ~0.35 mm squash per rim and a ~33 Hz vertical bounce (`contact_timeconst`
~5 ms). That made the simulated robot hop far more than the real one does, so the preset uses 10 ms.

### The worm drive (`motor worm`, `drive.py`)

A plain-PWM DC motor behind a self-locking worm gear. The motor has its own speed state and follows
the linear curve τ = τ_stall·(u·V/V_nom) − (τ_stall/ω₀)·ω, with Coulomb friction giving a ~5 % duty
deadband. When the motor drives the wheel, it feels the load. When the wheel would drive the motor
(braking, being pushed, coasting), the worm locks: the wheel is held to the motor's speed, and the
motor feels nothing (`backdrive_efficiency 0`). `pwm_decay` picks what the driver does in PWM
off-time: `brake` (motor shorted) or `coast`. `stop_mode` picks what duty 0 does. With the TB6612FNG
driven the usual way (IN1/IN2 set the direction, PWM on the PWM pin), the off-time is a short brake.
A zero command brakes if the direction pins stay set, and coasts if the firmware sets IN1 = IN2 = low. Step it with
`Drive(m, p).step(d, duty)` instead of `mj_step`. `gear_backlash` exists but is experimental: the
free play lets the robot rock on its 12 contacts, and flank impacts clunk harder than real gears.
Without backlash the gear holds both ways, with a stiffness of 1 N·m/rad (`coupling_kv` × `_KP`).
A one-sided flank and 4 N·m/rad made a resting robot fall into a ~22 Hz limit cycle (wheels ±0.25
rad/s, rollers up to 18 rad/s, yaw drifting −4° in 20 s); now it stays still.

### Results (`python test_drive.py --preset vsss`)

| test | result |
|---|---|
| open-loop duty from the no-load curve, 0.5 m/s along x / y | 81–82 % of commanded (gearbox friction ~8 %, rolling losses over the 12 contacts ~10 %) |
| spin 6 rad/s / arc | 74 % / 70–91 % |
| full duty along x | 0.76 m/s, 0→90 % in 138 ms, peak 9 m/s², 3° pitch (too fast: contact artefact, see above) |
| sudden stop from 0.76 m/s, `stop_mode brake` | stops in 43 mm, ~6° pitch (~5 mm side lift) |
| sudden stop, `stop_mode coast` | rolls ~21 cm, ~1° (~0 mm lift) |
| 200 ms ramp (either stop mode) | 65 mm, ~3° (~2 mm lift) (the robot outruns its wheels: contact artefact) |
| push the unpowered robot until it slides | 1.74 N along x, 1.52 N along a diagonal |

Pushing checks the friction anisotropy. With the wheels locked, each wheel resists only along its
drive direction (its rollers roll freely along the axle). That predicts 0.707·μ·m·g = 1.60 N along
the body axes and 0.5·μ·m·g = 1.13 N along the diagonals, where two wheels just roll. The same
holds for traction: an X-layout robot accelerates at most at ~0.71·μ·g along its axes and ~0.5·μ·g
along diagonals.

Vibration peaks around 0.4–0.5 m/s, where the 12 contacts per wheel revolution come at ~40 Hz,
close to the rims' bounce. It's 0.5–1.3 g rms depending on contact softness; compare with the IMU.
The hub never touched the floor in these runs.

### Estimates, and how to measure them

- **`kinematic_radius` (16.9 mm):** the geometric 17.06 mm minus about a third of the ~0.35 mm
  silicone squash. Measure distance per wheel revolution from video of a marked wheel. Without
  encoders, what matters in practice is the duty → speed map: with the camera working, drive
  constant duties and fit speed against duty (this lumps radius, friction, deadband and voltage).
- **`friction` (1.0):** soft silicone on painted MDF is usually ~0.8–1.2; randomize 0.7–1.3 until
  measured. The robot never tipping on hard stops implies μ < ~1.36 at a 25 mm COM. To measure,
  put the unpowered robot on a tilted MDF board, diagonal pointing downhill, and find the angle θ
  where it slides: μ = 2·tan θ (along a body axis: μ = √2·tan θ). This works because the worms lock.
- **`wheel_armature` (rotor inertia):** spin-up time of a lifted wheel at full duty, filmed.
  τ ≈ J·ω₀/τ_stall ≈ 35 ms for the current guess.
- **`contact_timeconst` / `contact_dampratio`:** IMU vibration against speed.
- **`com_height`:** balance the robot on an edge.

## How the wheel model works

It uses the same structure as the Gazebo URDFs:

```
chassis (free body)
 └─ wheel_i   hinge, driven. Axis points inward along the radius, at angle θ_i = offset + 360°·i/N
     └─ roller_i_kj   hinge, passive. Axis is tangent to the wheel. rows × rollers_per_row per wheel
```

By default the rollers are ellipsoids whose length is tuned so the wheel's rolling radius stays
within about 0.1 mm of `wheel_radius`. Ellipsoid–plane contact is analytic and smooth; a coarse
faceted mesh roller caused fake vibration at speed. `--roller_shape mesh` gives the exact barrel
profile with blunt ends. `--roller_shape peanut` gives figure-8 rollers, two lobes on one axle
that touch the floor instead of the waist. Roller length is always capped so rollers in the same
row keep `roller_end_gap` between them, or you can set it with `roller_half_length`
(`lobe_offset` / `lobe_half_length` for peanuts).

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
- `rows`, `rollers_per_row`, `roller_radius`, `row_spacing`, `roller_half_length`,
  `roller_end_gap` (from the wheel you buy or print; `rows 1` for thin single-row wheels)
- peanut rollers: `roller_radius` = lobe radius, `lobe_offset` = roller middle to lobe centre,
  `lobe_half_length` = lobe semi-length along the axle (0 = round lobes)
- `body_mass`, `com_height`, `hub_mass`, `roller_mass`
- motor, measured at the wheel after the gearbox: `stall_torque`, `no_load_speed`,
  `nominal_voltage`, `wheel_armature` (rotor inertia × gear²)
- `friction` (roller material on the field surface), `contact_timeconst`

`--motor servo` is an ideal speed loop with a torque cap. It is closest to what the Gazebo sim
does, which is ideal velocity joints with no motor model. `--motor dc` is a voltage-driven
linear DC motor, τ = (τ_stall/V)·u − (τ_stall/ω₀)·ω, which exposes torque and traction limits.

## Validation (defaults)

The numbers from here down were measured before the switch to elliptic friction cones
(see Notes). The generic servo results were re-checked afterwards and are unchanged; sim speed
dropped to ~8× real time.

| test | servo | dc (open-loop voltage) |
|---|---|---|
| vx / vy / diagonal tracking | 100 % | 99.6–99.9 % |
| spin 6 rad/s | 99.9 % | 99.6 % |
| arc (0.3 m/s + 3 rad/s) | 100 % | 99–100 % |
| speed | ~15× real time (now ~8×) | ~15× real time (now ~8×) |

A full-throttle launch with the placeholder DC motors is traction-limited. The motors can push
about 11× more force than friction allows (μ·m·g ≈ 1.5 N). The wheels spin, encoder odometry is off by more than 1 m/s,
and the robot chatters. Limit acceleration to about μ·g in firmware.

## Thin, single-row wheels

With one row, neighbouring rollers are only 360°/n apart, so they have to be short and the
wheel has real dips at the gaps. Example: a 32 mm wheel with 10 rollers of Ø6 mm.

| roller ends | rolling radius | distance vs nominal r | vertical vibration at 1 m/s |
|---|---|---|---|
| pointy (ellipsoid) | 15.42–16.00 mm | 98 % | ~0.9 g rms, airborne ~10 % of the time |
| blunt (mesh) | 15.85–16.00 mm | 96–98 % | ~0.9 g rms, airborne ~25 % of the time |
| (2×3 double row, for comparison) | 16.00–16.08 mm | 100 % | ~0.3 g rms, never airborne |

- Calibrate `r` on the real robot (distance per wheel revolution) rather than using the
  nominal diameter.
- The vibration level depends on how soft the contact is (`contact_timeconst`, standing in for
  rubber and chassis compliance). Tune it until the sim matches what the real robot does.
- The roller end shape matters here, so your real roller profile is better than either
  approximation.

### Peanut / spool rollers

For hourglass rollers whose rounded end rims touch the floor and whose concave waist doesn't.
Each roller is modelled as its two rims, flattened ellipsoids of radius `roller_radius` and
thickness `2 * lobe_half_length` (default `0.5 * roller_radius`), on one passive axle. The rim tips sit on
the rolling circle, and by default the 2n rims of a row are spaced evenly around the wheel. The
waist never touches, so it's left out. A spool is not convex; if you import one as a single mesh,
MuJoCo fills in the waist and the middle touches the floor. So keep it as two convex pieces.

Example: a 6-roller single-row wheel, Ø32 mm, with proportions measured from a photo of a real design
(`--roller_shape peanut --rows 1 --rollers_per_row 6 --roller_radius 0.00314`):

- geometry: roller 9.8 mm long, contact rims at ±15° (12 per revolution, the "dodecagon"),
  rolling radius 15.33–16.01 mm, effective radius 15.83 mm (98.9 % of the max)
- tracking with the max radius in the Jacobian: 97.8–99.3 %, which matches r_eff/r. Use
  `r_eff` (printed by `omni_mjcf.py` / `test_drive.py`) or a measured value as `r` in the kinematics.
- vertical vibration at 0.5–1.3 m/s: 0.85–1.1 g rms, airborne 8–25 % of the time with
  `contact_timeconst 0.01`; 0.4–0.8 g rms, airborne ≤ 4 % with `0.02`. Calibrate against the real robot.

## Notes

- Friction uses elliptic cones with `impratio 10`. MuJoCo's default pyramidal cone caps friction at
  μ/√2 along the diagonals of the contact frame, which is exactly where X-layout wheels slip.
  `impratio` > 1 stops soft friction from creeping under steady sideways loads.
- For `motor worm`, `coupling_kv` must stay ≤ `wheel_armature / timestep` (checked), and
  `contact_timeconst` ≥ 2 × `timestep`.
- Robot geoms only collide with the floor (and other robots' chassis). Gazebo's `selfCollide`
  on rollers is not reproduced.
- The repo's collision meshes can't be reused as-is. MuJoCo convex-hulls meshes, and the
  chassis hull swallows the wheels. `import_repo_urdf.py` shows this and works around it.
- `src/kinematics.cpp` hard-codes `ROBOT_RADIUS 0.088` (from the original 3w model). The
  3w_v2/4w/5w/6w wheels sit at 0.1028 m, so those robots turn at about 87 % of the commanded
  ωz. `import_repo_urdf.py` reproduces this.
