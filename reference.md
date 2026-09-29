# Project reference (handover)

Context for anyone, human or agent, picking this project up. `README.md` documents the original
Gazebo package; `mujoco/README.md` documents how to use the MuJoCo model. This file records the
goal, the real robot, what was built and why, what was verified, the traps found, and what's open.

## 0. Quickstart

Every command below was run on the owner's laptop (Ubuntu 22.04, ROS 2 Humble), where everything
is already installed. Use `python3`: Ubuntu 22.04 has no `python`.

**One-time setup** (a fresh Ubuntu 22.04 + ROS 2 Humble machine):

```bash
git clone https://github.com/arepo90/ros2_omni_robot_sim.git ~/ros2_omni_robot_sim
cd ~/ros2_omni_robot_sim && git checkout humble
pip install --user "mujoco==3.14.0" "numpy<2" python-xlib   # numpy < 2: Humble's rclpy and cv_bridge need 1.x
pip install --user torch==2.8.0 torchvision==0.23.0 ultralytics "numpy<2"   # the vision's YOLO (installed on the laptop)
sudo apt install ros-humble-teleop-twist-keyboard ros-humble-rqt-image-view ros-humble-rqt-plot
source /opt/ros/humble/setup.bash && colcon build --base-paths msgs   # vsss_msgs: the /field messages
echo 'source ~/ros2_omni_robot_sim/install/setup.bash' >> ~/.bashrc
# the YOLO robot detector from the old vision (vsss repo history; not in git, 52 MB):
git clone https://github.com/arepo90/vsss.git /tmp/vsss && git -C /tmp/vsss show \
  '7490988:bullet_ws/install/vision/share/vision/utils/models/yolov8m(v1)/best.pt' > vision/models/robots_yolov8m.pt
```

Every terminal needs ROS and `install/setup.bash` sourced. On the laptop `~/.bashrc` does both
(plus `ROS_DOMAIN_ID=20`, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`). Without `vsss_msgs` the sim still
runs, just without `/field_truth`.

**Demos without ROS** (from the `mujoco/` folder, so the generated `.xml` files stay gitignored):

```bash
cd ~/ros2_omni_robot_sim/mujoco
python3 test_drive.py --preset vsss              # roller model checks: tracking, launch, stops, push (~5 s)
python3 planar_robot.py                          # fast model vs roller model, side by side (~10 s)
python3 omni_mjcf.py --preset vsss               # writes vsss_omni4.xml: one robot, roller model
python3 -m mujoco.viewer --mjcf=vsss_omni4.xml   # look at it (the wheel detail; nothing drives it here)
python3 vsss_field.py --blue 3 --yellow 3        # writes vsss_field.xml: field, ball, 6 robots
python3 -m mujoco.viewer --mjcf=vsss_field.xml   # look at it (nothing drives the robots here)
python3 import_repo_urdf.py 4w                   # the original Gazebo 4w robot in MuJoCo (needs ROS sourced)
```

**The ROS 2 sim**, three terminals:

```bash
# terminal 1: the sim, ball, overhead camera, 3D viewer. Stop: Ctrl+C or close the viewer. Pick one:
cd ~/ros2_omni_robot_sim/mujoco
python3 ros_bridge.py --model roller --blue 1   # full model: real wheels and rollers on contacts (real time up to 3 robots)
python3 ros_bridge.py                           # fast model (the default): 3 blue robots, 3v3 with --yellow 3

# terminal 2: drive blue_0 with the keyboard; it moves only while keys are held and this terminal has focus
cd ~/ros2_omni_robot_sim/mujoco
python3 teleop.py                      # --robot yellow_1 for another robot, --speed / --turn for 100 %

# terminal 3: watch
ros2 run rqt_image_view rqt_image_view /overhead_camera/image_raw   # the overhead camera
ros2 run rqt_plot rqt_plot /blue_0/odom/twist/twist/linear/x         # plot any field
ros2 topic echo /blue_0/odom                                          # ground-truth pose and velocity
```

Teleop keys (`teleop.py`): `W` / `S` forward / back, `A` / `D` left / right, `Q` / `E` turn counter-clockwise /
clockwise, `↑` / `↓` speed ±10 %, `Esc` or Ctrl+C quit. Keys combine (W+A diagonal, W+Q arc). 100 % is
0.8 m/s and 6 rad/s; it starts at 50 %. The robot moves only while a key is held: releasing sends a
stop at once (a 100 ms tap moves it ~2 cm), and so does switching to another window. It reads the
physically held keys from X11 (not Wayland). `ros2 run teleop_twist_keyboard teleop_twist_keyboard
--ros-args -r cmd_vel:=/blue_0/cmd_vel` also works, but its commands stick until the next key.

**Vision** (§9): camera image → `/field`, the global state strategies consume. With the sim running:

```bash
python3 vision/vision_node.py                                       # sim camera; pose from TF + camera_info
ros2 run rqt_image_view rqt_image_view /vision/image_annotated      # what it sees: ids, headings, ball
ros2 topic echo /field                                              # the state (ros2 topic echo /field_truth: the truth)
python3 vision/vision_eval.py                                       # accuracy vs the sim's truth, every 10 s
# a real camera: click 6 line points once (camera fixed), then run on its topic
python3 vision/calibrate.py --device 2 --camera_height 1.95 --colors -o vision/calib/real.yaml
python3 vision/vision_node.py --calib vision/calib/real.yaml --image /camera/image_raw
```

Without teleop, from any terminal:

```bash
ros2 topic pub --once /blue_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"   # drive; holds until the next command
ros2 topic pub --once /blue_0/cmd_vel geometry_msgs/msg/Twist "{}"                    # stop
ros2 service call /reset std_srvs/srv/Empty                                           # robots and ball back to the start
```

Useful `ros_bridge.py` options (`python3 ros_bridge.py -h` lists them all):

| option | effect |
|---|---|
| `--blue 3 --yellow 3` | robots per team, 0–5 (`blue_0`…, `yellow_0`…) |
| `--no_viewer` | headless: no 3D window, topics and camera still run |
| `--model roller` | the full roller-level model (real time for 1–3 robots; 3v3 runs at ~0.8×) |
| `--cam_ortho` | flat, parallax-free camera instead of a pinhole at 2 m |
| `--cmd_timeout 0.5` | stop a robot 0.5 s after its last command (default: hold) |
| `--ff_gain 1.2` | scale the open-loop duty: robots reach only ~80 % of the command |

Two robot models, same topics and controls. **roller** is the full simulation: every roller is a
body on its own axle touching the floor, driven through the worm-gear model (`drive.py`); it has
pitch, bounce and vibration, and is the reference. **planar** (default) is a box pushed by one
traction force per wheel with the same motor and self-locking behaviour, fitted to the roller
model's steady-state driving (§5.8); ~20× real time for a full field, for 3v3 and future RL.

In the 3D viewer the cases are see-through grey so the wheels show (the planar model's wheels are
for display only, turning with its wheel speed; they're left out without a viewer). The overhead
camera still sees opaque black cases like the real robots. Double-click a robot or the ball, then
Ctrl + right-drag to push it. All topics are listed in `mujoco/README.md`.

If something is off:

- **`ros2 topic list` shows only `/rosout` and `/parameter_events`:** the sim isn't running, or the
  Zenoh router is down (`systemctl status rmw-zenoh-router`; it's a system service and must stay
  on, since `RMW_IMPLEMENTATION=rmw_zenoh_cpp` needs it for discovery).
- **The robot ignores teleop:** click the teleop terminal (its status line says "not focused"
  otherwise) and check the robot name (`ros2 topic echo /blue_0/cmd_vel` prints while you hold keys).
- **Robots are slower than commanded:** expected. The duty is open loop, like the real firmware.
- **The RoboCorea services** (`camera-streamer`, `esp32-bridge`, `robot-manager`, `map-manager`,
  `c920-stream`) are disabled so the ROS graph only holds the sim. To bring them back:
  `systemctl --user enable --now camera-streamer esp32-bridge robot-manager map-manager c920-stream`.

## 1. Goal

The owner is building a fleet of 4-wheel omnidirectional robots for **IEEE VSSS** (Very Small Size
Soccer; robots fit a 7.5 cm cube). Over a 4-month plan they want to train cooperative multi-agent RL
policies in **MuJoCo** and transfer them to the real fleet. Main points of the plan:

- **Perception:** an overhead camera (≥ 60 Hz) and a Kalman-filter state tracker give positions,
  headings and velocities of the ball and all robots.
- **Policy:** outputs body velocities (vx, vy, ω) per robot, sent over radio. The actor may only use
  Conv1d/Conv2d, Linear, and ReLU/SiLU: no RNNs, attention, or norm layers with running stats.
  Temporal context comes from frame stacking.
- **Training:** MAPPO with centralized training and decentralized execution (centralized critic,
  decentralized actors). Curriculum: single-robot skills → 2v0/3v0 → self-play 3v3/4v4.
- **Sim-to-real:** system identification of wheels, motors, friction and the ball; latency of
  30–90 ms injected with buffer queues; domain randomization of mass/COM ±5 %, friction ±15 %,
  actuator gains ±10 %, latency jitter, and sensor noise; software velocity/acceleration limiters.
- **Timeline:** month 1 is the MuJoCo model, sysID and the vision pipeline; month 2 low-level control
  and single-agent RL; month 3 MARL; month 4 fleet deployment.

This repo is an unedited fork of `YePeOn7/ros2_omni_robot_sim` (Gazebo Ignition Fortress, ROS 2
Humble). The owner cares about the **Humble** branch (`humble`) and mainly wants its omni-wheel
model reproduced in MuJoCo, sized to their robot. All new work is in `mujoco/`. The Gazebo package
itself is untouched.

## 2. The real robot

Given by the owner (measured or from datasheets), with derived values in the model:

| item | value | model (`--preset vsss` in `mujoco/omni_mjcf.py`) |
|---|---|---|
| base | 7.5 × 7.5 cm, height ≤ 7.5 cm | `body_size 0.075` |
| layout | 4 wheels, axles along the base diagonals (X layout), wheel centres 34 mm from the base centre | `heading_offset_deg -45`, `wheel_R 0.034` |
| mass | ~230 g total | `body_mass 0.214` + ~4 g/wheel (estimated) |
| COM | battery and motors at the bottom; owner guessed ~40 mm "maybe less"; 25 mm matches their observations | `com_height 0.025` |
| wheel | single plate, 6 rollers in one row; whole wheel inside Ø34.5 mm, hub inside Ø31.4 mm (PLA) | `wheel_radius 0.01725`, `hub_radius 0.0157` |
| rollers | spool / hourglass shape: 9.8 mm long, Ø6.7 mm at the ends, Ø5 mm in the middle; the end rims touch the floor, not the waist; cast silicone, a bit softer than Shore A40 | `roller_shape peanut`, `rows 1`, `rollers_per_row 6`, `roller_radius 0.00335`, `lobe_half_length 0.00128` |
| motors | 4× GA12-N20 12 V 381 rpm, long version with a 90° worm output; stall 700 mA / 235 gf·cm, no-load 30 mA; rated torque 71 gf·cm | `motor worm`, `nominal_voltage 12`, `no_load_speed 39.9`, `stall_torque 0.0230`, `wheel_frictionloss 0.001` (from I₀/I_stall) |
| gearbox | worm, self-locking (confirmed: a wheel won't turn by hand) | `backdrive_efficiency 0` |
| motor driver | 2× TB6612FNG. Usual wiring (IN1/IN2 direction, PWM pin) short-brakes in PWM off-time; duty 0 brakes if the direction pins stay set, coasts if IN1 = IN2 = low (firmware's choice) | `pwm_decay brake`, `stop_mode brake` / `coast` |
| battery | 3S LiPo 2000 mAh | `supply_voltage 11.1` (12.6 full, ~10.5 empty) |
| control | plain PWM; no encoders, no current sensing; firmware uses accel/decel ramps | commands are duty cycles |
| field | painted MDF, rollers "pretty grippy" | `friction 1.0` (estimate: 0.8–1.2 typical; never tipping implies < ~1.36) |
| rolling radius for commands | not measured | `kinematic_radius 0.0169` (17.06 mm geometric minus ~1/3 of the ~0.35 mm squash) |

Owner's observation: before ramps, sudden starts and stops lifted one side of the robot by "a couple
of mm", and it never came close to tipping. Ramps fixed it. This was used to pick `com_height` and
`contact_timeconst`.

Photos of the wheel (side and top views) showed the rims touching at about ±17° from each roller
centre, 12 contacts per revolution ("dodecagon"), about 26° gaps between rollers filled by hub
spokes that reach ~93 % of the radius, and a hub about 3.4 mm thick.

## 3. Repository map

```
src/kinematics.cpp          Gazebo-side kinematics/odometry node (ROS 2); has bugs, see §4
urdf/{3w,3w_v2,4w,5w,6w}/   robots; 3w_v2/4w/5w/6w share the same wheel and roller meshes
config/controller_configs/  ros2_control: one JointGroupVelocityController per wheel, 50 Hz
launch/                     gazebo_sim / slam / navigation launch files (OMNI_ROBOT_MODEL env var)
mujoco/omni_mjcf.py         parametric MJCF generator + wheel Jacobian + presets ("generic", "vsss")
mujoco/drive.py             Drive(m, p).step(d, u): servo / dc / worm-gear motor models; apply()/update()
                            around one shared mj_step for several robots (name prefix)
mujoco/test_drive.py        validation: tracking, launch, sudden/ramped stop, push tests
mujoco/planar_robot.py      fast planar robot (box + one traction force per wheel) + side-by-side test
mujoco/vsss_field.py        VSSS field (walls, goals, lines), ball, N robots (planar/roller), camera, via MjSpec
mujoco/ros_bridge.py        rclpy bridge: <robot>/cmd_vel in; odom, imu, wheels, duty, camera, tf, clock out
mujoco/teleop.py            terminal keyboard teleop (WASD/QE, hold to move) for one robot's cmd_vel
msgs/vsss_msgs/             Field / Object messages: the global state (/field, /field_truth); colcon build
vision/patterns.yaml        robot top patterns (colours, id table, layout): shared by the sim and the vision
vision/vision_node.py       camera -> /field: YOLO + colour patches + Kalman filters (camera, detect, track .py)
vision/calibrate.py         real camera: click 6 field line points (+ patch colours) -> calibration YAML
vision/vision_eval.py       scores /field against the sim's /field_truth
mujoco/import_repo_urdf.py  1:1 import of the Gazebo URDFs into MuJoCo (needs `pip install xacro`)
mujoco/README.md            usage and parameter documentation
```

Setup and every command to run: §0. Tested with Python 3.11, MuJoCo 3.14.0, NumPy 2.4, and on
Ubuntu 22.04 / ROS 2 Humble with the system Python 3.10 and NumPy 1.26: identical results. The
Gazebo part needs ROS 2 Humble, Gazebo Fortress, `install_dependency.sh`, and `colcon build` (see
`README.md`). No GPU is needed for the sim; the vision's YOLO uses one if present (CPU works, slower).

## 4. How the original Gazebo sim works, and its bugs

- **Wheel:** a Ø58 mm hub (`omni_frame.stl`) on a continuous joint, axis `0 0 -1` in the wheel
  frame, which is radially inward on the robot. It carries 6 passive rollers in 2 staggered rows of
  3: Ø18 × 33 mm barrels whose profile follows a 30 mm circle, axes 21 mm from the hub, rows at
  z = 2.5 and 17.5 mm. The wheel is round within 0.04 mm (measured from the STLs). The hub clears
  the ground by only 1 mm.
- **Actuation:** ideal velocity joints through `ign_ros2_control`, at 50 Hz. There's no motor model
  and no torque limit, so the sim says nothing about acceleration or traction.
- **Masses and inertias:** mostly placeholders. The chassis has ixx = iyy = 0.9 kg·m², about 50×
  a real 3 kg, 25 cm box; wheels are 0.005 kg·m², about 20× too large.
- **Kinematics** (`src/kinematics.cpp`): wheel i sits at θᵢ = 360°·i/N + offset (4w: −45°), and
  ωᵢ = (−sin θᵢ·vx + cos θᵢ·vy + R·ω)/r. Odometry takes translation from the pseudo-inverse and
  heading from the IMU. The math is general and scales to any size.
- **Bugs:**
  - `ROBOT_RADIUS 0.088` is right only for the original 3w robot. In 3w_v2 (the default), 4w, 5w
    and 6w the wheel mid-plane sits at 0.1028 m, so those robots turn at ~87 % of the commanded ω.
    This was verified by importing them into MuJoCo (`import_repo_urdf.py`), where R = 0.1028 gives
    99–101 %.
  - `/odom` twist `vx`/`vy` are never computed and are always 0.
  - `mOd` is dead code.
  - `360 / N` uses integer division; harmless for N = 3..6.
  - Not fixed; a fix was proposed as a separate task.

## 5. The MuJoCo model: decisions and why

Each decision was tested; see `mujoco/README.md` for numbers.

1. **Roller-level model.** Chassis (free body) → wheel hinge (driven, axis inward) → rollers on
   passive hinges (axis tangent). Same structure as the URDFs, generated from parameters instead
   of meshes.
2. **Collision filtering.** MuJoCo replaces every mesh with its convex hull. The Gazebo chassis
   hull swallowed all 24 rollers and the imported robot couldn't move. Robot geoms therefore collide
   only with the floor (bitmasks: floor 1/1, rollers and hub 2/1, chassis 4/5 so chassis collide
   with each other and the floor, visuals 0/0). Roller self-collision is off.
3. **Roller shapes.**
   - **ellipsoid:** length tuned for roundness. Smooth analytic contact, fastest.
   - **mesh:** the exact barrel profile. A coarse mesh (9×16) caused fake vibration and the robot
     was airborne 13 % of the time at 1 m/s; 17×32 is fine.
   - **peanut:** the owner's spool rollers, modelled as two flattened ellipsoid rims per roller on
     one axle, evenly spaced (30° for 6 rollers). A spool is not convex, so as a single mesh it
     would be hulled into a barrel whose middle touches the floor.
   - Roller length is always capped so rollers in the same row keep `roller_end_gap`. Without the
     cap, single-row wheels overlapped their neighbours and looked unrealistically smooth.
4. **Effective rolling radius** `r_eff` = mean of the rolling-radius profile = distance per
   revolution / 2π. It's printed by the tools. The sim reproduces commanded motion to ~1 % when the
   Jacobian uses r_eff. For the owner's wheel it's 17.06 mm against 17.25 mm nominal. Use r_eff, or
   better a measured value, in firmware.
5. **Friction.**
   - MuJoCo uses the larger of the two geoms' friction values and the floor defaults to 1.0, so
     rollers set `priority="1"` to make their value count. (This was a bug in the first version:
     changing friction did nothing.)
   - The **elliptic** cone is used because the default pyramid caps friction at μ/√2 along the
     contact frame's diagonals, which is exactly where X-layout wheels slip.
   - **`impratio 10`**, because soft friction otherwise creeps: an unpowered, locked robot slid at
     12 mm/s under 0.8 N.
6. **Contact softness** (`contact_timeconst`, damping ratio 1). The Hertz estimate for the silicone
   rims is ~5 ms, but that made the robot hop a lot at speed (resonance of the 12 contacts at
   ~40 Hz near 0.5 m/s), and locked wheels landing on the floor kicked it into big pitch spikes.
   10 ms matches the owner's observations much better. MuJoCo's static sink is also much smaller
   than the physical squash, so calibrate r from measurements.
7. **Motor models** (`motor`):
   - **servo:** ideal speed loop with a torque cap, like Gazebo.
   - **dc:** linear motor curve, ctrl = volts.
   - **worm:** the owner's drive; details in `mujoco/README.md` and `drive.py`. The motor speed is
     a state; power flowing motor→wheel is transmitted with full load; wheel→motor locks
     (`backdrive_efficiency 0`). `pwm_decay` is `brake` or `coast`. Coulomb friction gives a ~5 %
     duty deadband.
   - Stability constraints: `coupling_kv` ≤ `wheel_armature / timestep` (the explicit coupling
     diverged otherwise; checked in code), and `contact_timeconst` ≥ 2 × `timestep` (0.5 ms).
   - `gear_backlash` is experimental. It caused rocking on the 12 contacts and exaggerated flank
     impacts, so it's 0 in the preset.
   - Without backlash the gear holds both ways (both flanks engaged) with stiffness 1 N·m/rad
     (`coupling_kv` × `_KP` 50). The first version kept the one-sided flank logic at zero backlash
     and used 4 N·m/rad; a resting robot then fell into a ~22 Hz limit cycle (wheels ±0.25 rad/s,
     rollers up to 18 rad/s, yaw −4.3° in 20 s). Both changes were needed; raising roller damping
     instead also stopped it but made a sudden stop pitch 29°. Tracking and launch didn't change.
8. **Fast planar model** (`planar_robot.py`, the default in the field and bridge). A box on
   slide x / slide y / hinge yaw joints; each wheel is a site actuator at its contact point pushing
   along the rolling direction with k·(r_eff·ω − v_t), capped at μ·N (N = mg/4, no load transfer),
   free along the axle. Same DC motor curve, rotor inertia, gearbox friction and PWM modes as
   `drive.py`.
   - Self-locking is one guard, not a gear model: the load the wheel puts on the motor is clamped
     so the ground can resist the motor but never drive it. The motor speed is solved with backward
     Euler each step (three closed-form cases: unloaded, loaded and gripping, loaded and slipping),
     using the same v_t MuJoCo then uses for the traction force (`mj_step1` / `mj_step2`). Results
     agree to < 1 mm between 0.5 and 4 ms steps, robot kinetic energy matches the traction work to
     1 %, and robots at duty 0 don't move (0.00 µm in 10 s after 60 s of random 3v3 driving).
     A first version carried a "driving or back-driven" state between steps and needed an implicit
     load term plus a sign-flip patch; this replaces both.
   - Traction is integrated explicitly (semi-implicit Euler) at 2 ms: the implicit integrators keep
     the velocity derivative of a force-capped actuator, which acted like extra mass while the
     wheels slipped (6.1 instead of 6.9 m/s² at the traction limit).
   - Rolling losses (`rolling_resistance` 0.03, `rolling_damping` 6e-5 N·m·s/rad) are part of the
     clamped load, fitted to the roller model's steady-state tracking; roller drag and yaw scrub
     turned out unnecessary (0). Mass and yaw inertia come from the roller model.
   - Dropped: pitch/roll, load transfer, the 12-contact vibration. ~30× real time per robot, ~20×
     for a full field (fixed ~95 µs per step, nearly independent of robot count).
   - Looks: in the viewer the case is a see-through grey shell (render group 2) and each wheel is a
     display-only body on a hinge that `PlanarDrive` sets to the wheel angle (no collisions, 1e-5 kg);
     about +30 % per step, so the bridge adds them only with the viewer. Cameras rendered with
     `vsss_field.camera_option()` see the opaque black collision box (group 3) instead.

## 6. Key results for the owner's robot (preset `vsss`)

- **Top speed:** 0.76 m/s along a body axis at 11.1 V full duty. Scale by battery voltage, which
  runs 10.5–12.6 V over a charge (about ±7 %).
- **Open-loop duty feed-forward:** gives ~81–82 % of the commanded speed (gearbox friction, rolling
  losses over the 12 contacts, scrub between locked worms). Speed control must close the loop
  through the camera, or the feed-forward must be calibrated.
- **Traction limits** (X layout): ≈ 0.71·μ·g along the body axes and 0.5·μ·g along the diagonals.
  The motors (stall 0.023 N·m) exceed this at low speed, so full-duty starts spin the wheels a little.
- **Stops and pitch** (after the idle fix, §5.7):
  - sudden stop from full speed with `stop_mode brake`: ~43 mm, ~6° pitch (~5 mm side lift);
    before the fix 35 mm, 10°, 8 mm;
  - with `stop_mode coast`: rolls ~21 cm with ~0 mm lift;
  - 200 ms ramp (either mode, since the TB6612 brakes in PWM off-time): 65 mm, ~2 mm lift.
  - The owner saw "a couple of mm" without ramps, which falls between the two; rotor inertia is
    the main unknown, plus which stop mode the firmware used.
- **Being pushed:** the unpowered robot resists 1.74 N along a body axis and 1.52 N along a
  diagonal (predictions 1.60 / 1.13 N at μ = 1). Pushed robots skid rather than roll.
- **Roller-model transients are suspect.** On a full-duty launch the robot gains more kinetic
  energy than the gears deliver to the wheels (3.1 vs 1.6 mJ at 20 ms, 36.5 vs 30.2 mJ at
  100 ms) and exceeds the 6.9 m/s² traction limit (peak 8.9); the slipping rims bounce on the soft
  contacts (contact count 0–6). In the 200 ms ramp stop the robot stops, then rolls back at
  0.2 m/s, while its wheels still turn forward. So the 138 ms launch and the ramp-stop distance
  are artefacts; the planar model (204 ms, 109 mm) follows the motor model. Steady-state results
  are consistent (energy balances, planar fit matches).
- **Vibration:** peaks around 0.4–0.5 m/s at 0.5–1.3 g rms, depending on contact softness.
- **Sim speed:** roller model ~6–8× real time for one robot; on the field 4.7× with 1 robot, 2.0×
  with 3, 1.08× for 3v3 (bridge with viewer 0.8×). Planar model ~20× for a full field; the bridge
  holds real time for 3v3 with viewer and camera. Still far too slow for large-scale MARL self-play:
  that needs the planar equations batched (numpy/torch/JAX over many environments) or MJX.
- **Idle:** both models now stay still at duty 0 (60 s: no drift). The roller model used to fall
  into a ~22 Hz limit cycle, fixed in `drive.py` (§5.7); `test_drive.py` settles for only
  0.3–0.5 s, so it never showed up.

## 7. Open items

Values to measure (procedures are in `mujoco/README.md`, "Still placeholders"):

- **`friction`:** estimated 1.0; incline test, μ = 2·tan θ with the diagonal pointing downhill.
- **Kinematic radius:** estimated 16.9 mm; distance per wheel revolution from video. In practice
  fit the duty → speed map with the overhead camera instead.
- **`stop_mode`:** which one the firmware uses (brake if IN1/IN2 stay set at duty 0, coast if both
  go low).
- **Driver losses:** TB6612FNG on-resistance (roughly 0.5 Ω against ~17 Ω of motor) costs a few
  percent of voltage; fold it into `supply_voltage` or the feed-forward fit.
- **`wheel_armature`:** spin-up time of a lifted wheel on video.
- **`contact_timeconst` / `contact_dampratio`:** IMU vibration against speed.
- **`com_height`:** balance test.
- **Firmware ramp rates:** model them exactly in the environment.

Done since: the field, ball and multi-robot scene (`vsss_field.py`; dimensions from FIRASim's
Division B defaults, corner triangles 7 cm; still to check against the current rules; ball
rolling resistance is a guess), the ROS 2 bridge (`ros_bridge.py`), the idle limit cycle fix, the
fast planar model, and the owner's 3-colour robot tops (colours sampled from their pattern sheet;
since corrected: the ID squares are the front, as in the old vision code). Ball and robot-case
friction are 0.4 and 0.3 (guesses): with MuJoCo's default 1 on both the floor and the robot face, a
pushed ball jammed (it locks once mu_face * mu_floor >= 1) and the robot crawled at 2 cm/s behind it.
Also the stack of §9: `vsss_msgs`, `/field_truth`, the vision node with calibration and scoring.

Not done yet:

- **Vision on the real camera**: calibrate, sample colours, check the old YOLO weights on the
  current robots (§9). Ball: 5 mm error in the sim from the silhouette centroid.
- **Latency and ramps**: the radio delay and the firmware's ramp rates, measured, in the env and
  optionally in the bridge (`--ramp` already sets the duty ramp).
- **Roller-model transients** (§6): find why slipping rims inject energy (contact softness,
  `impratio`, timestep) before trusting its launch / stop / vibration numbers.
- The Gym/PettingZoo environment: action pipeline (vx, vy, ω) → ramp limiter → inverse
  kinematics with r_eff → duty feed-forward (plus deadband compensation) → latency queue →
  `Drive.step`. Observations should come from the simulated overhead tracker (no encoders on the
  real robot), with privileged state only for the critic. Also domain randomization of battery
  voltage, friction, mass/COM, motor gains and latency.
- Batch the planar model for MARL (vectorised numpy/torch/JAX across environments, or MJX), and
  fit its placeholders (rotor inertia, rolling losses, μ) to the real robot with the camera.
  Keep the roller model as the reference for geometry questions.
- The `src/kinematics.cpp` fixes from §4.

## 8. Conventions

- **Frames:** body x forward (red marker on the chassis top), z up. The freejoint has linear
  velocity in the world frame and angular velocity in the body frame. Wheel i (1-based) sits at
  angle `heading_offset_deg + 360·(i−1)/N`. The joint axis points inward, so positive speed means
  counter-clockwise tangential motion (same sign convention as `src/kinematics.cpp`).
- **Names:** joints `wheel_i`, `roller_i_kj` (k = row, j = index); actuators `motor_i`; sensors
  `imu_quat`, `imu_gyro`, `imu_acc`, `enc_i` (wheel joint velocity; the real robot has no encoders).
- **Presets:** `make_params("vsss", **overrides)`. Every field is also a CLI flag in
  `omni_mjcf.py` and `test_drive.py`, with defaults taken from `--preset`.
- **Model decisions:** keep them evidenced. Measure in the sim and compare against physics
  predictions or the real robot before changing defaults, and record the result in
  `mujoco/README.md` and here.

## 9. The stack: vision, `/field` and control

Same architecture as the owner's earlier system (github.com/arepo90/vsss: `vision` → `field_data` →
`strat` → `/low{i}` → sim or ESP32 bridge), with RL taking over the strategy:

```
MuJoCo sim ── /overhead_camera/image_raw ──► vision_node ── /field ──► strategy / RL policy ── /<robot>/cmd_vel ──► sim
(or a real camera)                                           ▲                                                    (or radio → ESP32)
MuJoCo sim ── /field_truth (same message, exact) ────────────┘ for scoring the vision, or training without it
```

- **Messages** (`vsss_msgs`, owner's choice: modelled on their `sim_msgs`, cleaned up). `Field`:
  header (stamp of the camera frame), `ball`, `blue[]`, `yellow[]`; `Object`: id (pattern id 1–10
  blue, 11–20 yellow; ball 0), `detected`, x, y, theta, vx, vy, w in the field frame (m, rad; origin
  at the centre, x towards the yellow goal). Absolute team colours: which team is "ours" is the
  strategy's business. Commands stay `geometry_msgs/Twist` on `/<robot>/cmd_vel` (body frame).
- **Patterns** (`vision/patterns.yaml`): the old vision's id table and the owner's pattern sheet.
  The front of a robot is the ID-colour half (the old code measured heading from the robot centre
  towards the ID squares, and its id table only matches the sheet that way); left/right are the
  robot's own. The sim paints its robots from the same file, robot blue_i = id i+1, yellow_i = 11+i.
- **Vision** (`vision/`), following the old YOLO vision (vsss commit 7490988, `vision_general.py`)
  and reusing its YOLOv8m weights, which find sim robots at 0.88–0.89 confidence unchanged:
  - YOLO runs in a background thread at up to 10 Hz (`--yolo_hz`) and finds robots; every frame
    the colour patches are decoded in a window around each tracked robot and each YOLO box no
    robot decoded in YOLO's own frame explains (comparing against tracks extrapolated back to that
    frame left a trailing extra window behind robots that had just started or stopped).
  - YOLO speed: torch 2.8.0+cu128 (2.0.1+cu117 had no kernels for the RTX 4060, sm_89, and took
    44 ms). `YoloRobots` runs the fused network in fp16 as a CUDA graph with one wait for the GPU and
    NMS in numpy: 12 ms and ~1 ms of CPU per run, against 23 ms through ultralytics' predictor; boxes
    within 0.6 px of it. Every blocking torch call releases the GIL and can wait the 5 ms switch
    interval to get it back, hence so few. `--yolo_hz 60` gives ~40 runs/s without slowing the
    frame loop; 10 is enough since tracks follow known robots. When upgrading torch: the old
    `nvidia-*-cu11` wheels overwrite same-named cu12 libraries (NCCL), so remove them.
  - Windows are as wide as a robot top on each side of the robot centre. At 0.8× the edge cut a
    touching robot's ID squares, their centroids moved inwards, and they could fit this robot's
    team patch better than its own pair: a nonexistent id about once per 3 minutes of 3v3. The
    tracker also won't open a track within 5 cm of another fresh one (robots are 7.5 cm wide).
  - Colours: a 64³ lookup table built from reference colours (nearest in CIE Lab, with background
    colours so dark/white pixels stay unlabelled): the old `lut_*.npy` idea, generated. Real
    cameras: sample the patches with `calibrate.py --colors`.
  - Geometry on the plane of the robot tops (7.05 cm), so parallax is removed (up to ~3 cm near the
    goals at 2 m) and heading / left-right don't depend on image axes. The two ID squares are the
    pair of ID-colour blobs that best fits the layout (35.5 mm apart, 35.5 mm from the team patch).
  - Kalman filters: constant velocity per robot id (x, y, heading) and for the ball, timed by frame
    stamps; a decoded id far from its track is ignored twice, then the track restarts there.
  - Results on the sim (3v3, every robot in random start/stop bursts like hold-to-move teleop,
    3 min at `--yolo_hz` 10 and 60 each): robots read with the right id 100 %, no wrong ids, no
    reads of ids that aren't on the field; position 0.3 mm mean / 0.8 mm 95 %, heading 0.3° / 1.2°,
    velocity 18 / 60 mm/s (robots at 0.1 / 0.5 m/s); ball found 100 %, 3–4 mm; 60 of 60 frames/s at
    3.0–3.3 ms each, exactly one window per robot; frames are 18–20 ms old when `/field` goes out.
    Offline on 40 random frames: 240/240 ids, 0.1 mm, 0.34°; 600 touching pairs: 600/600.
  - Calibration: the camera is fixed, so it's done once, like the old homography clicks, but it
    solves the full camera pose (6 line points, SQPnP; IPPE picked a wrong pose from straight above).
    Intrinsics from a checkerboard `camera_info`, or a lens guess plus the measured camera height:
    from straight above the floor can't separate focal length from height, and the height sets the
    parallax correction. 0.5 px click noise → 1.4 mm mean error (12 mm without the height). Through
    the node with such a file: 3.0 mm, 100 % ids.
- **Low-level control** (assessment of Guldner & Utkin 1995, sliding mode gradient tracking,
  `vsss/Sliding_mode_control_for_gradient_tracki-1.pdf`, which the old strategy built on): its
  controller needs force inputs and fast full-state feedback. The robot has PWM into self-locking
  worms (a velocity source with ~35 ms lag), no encoders, and 60 Hz camera feedback with 30–90 ms of
  latency; the switching law would chatter, and smoothing it leaves a saturated P controller. So:
  per-robot duty → speed calibration from the camera (removes the ~20 % open-loop shortfall and the
  deadband), firmware ramps at the traction limit, and optionally a slow (~1 Hz) integral trim on
  the vision velocity. Keep from the paper: the speed profile v = min(a₀t, v₀, √(2a₀d)) (time-optimal
  under an acceleration limit; stops without overshoot) for the software limiter, and its
  potentials as potential-based reward shaping for RL (γΦ(s') − Φ(s) keeps the optimal policy) and
  as a scripted baseline.
- **RL** does all the high-level control and strategy, consuming `/field` (or the same state inside
  the training env) and producing `cmd_vel`. The real robots have a radio delay, which forced accel
  / decel ramps in the firmware; in the sim starts and stops are nearly instant with the default
  `--ramp 0.2`. Both belong in the env (latency queue, the firmware's ramp rates).

## 10. History (branch `humble`)

- `1f6905d` MuJoCo port: generator, drive test, URDF importer.
- `2be0b4d` single-row wheels: roller length capped by neighbours.
- `d7260bf` spool/peanut rollers, effective rolling radius.
- `0f656a3` `vsss` preset for the real robot, hub collision, supply voltage.
- `c58d59b` worm-gear drive model (`drive.py`), elliptic cones + `impratio`, launch/stop/push
  tests, COM 25 mm, contact 10 ms, this file.
- `fc5fe35` owner's answers (self-locking confirmed, TB6612FNG, friction and radius estimates),
  `stop_mode`, `kinematic_radius`.
- `bc49197` VSSS field + ball + overhead camera (`vsss_field.py`), ROS 2 bridge
  (`ros_bridge.py`), multi-robot `Drive`, idle limit cycle fix, fast planar model with self-locking
  as a load guard (`planar_robot.py`), the owner's 3-colour robot tops, roller-model transient
  artefacts documented.
- this commit: teleop, quickstart, ball/case friction, see-through cases and display wheels,
  guard-based self-locking, `vsss_msgs` + `/field_truth`, pattern table (front = ID squares), the
  vision (YOLO + colour patches + Kalman, calibration, scoring; torch 2.8 with fp16 CUDA-graph
  YOLO, windows checked against YOLO's own frame and wide enough for touching robots), the
  low-level control assessment.
