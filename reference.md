# Project reference (handover)

Context for anyone, human or agent, picking this project up. `README.md` documents the original
Gazebo package; `mujoco/README.md` documents how to use the MuJoCo model. This file records the
goal, the real robot, what was built and why, what was verified, the traps found, and what's open.

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
| gearbox | worm, assumed self-locking (not yet confirmed by a hand-turn test) | `backdrive_efficiency 0` |
| battery | 3S LiPo 2000 mAh | `supply_voltage 11.1` (12.6 full, ~10.5 empty) |
| control | plain PWM; no encoders, no current sensing; firmware uses accel/decel ramps | commands are duty cycles |
| field | painted MDF | `friction 1.0` (placeholder) |

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
mujoco/drive.py             Drive(m, p).step(d, u): servo / dc / worm-gear motor models
mujoco/test_drive.py        validation: tracking, launch, sudden/ramped stop, push tests
mujoco/import_repo_urdf.py  1:1 import of the Gazebo URDFs into MuJoCo (needs `pip install xacro`)
mujoco/README.md            usage and parameter documentation
```

Setup: clone the repo and `git checkout humble`, then `pip install mujoco numpy` (and `xacro` for
the importer). Run `python mujoco/test_drive.py --preset vsss` and view with
`python -m mujoco.viewer --mjcf=vsss_omni4.xml` after `python mujoco/omni_mjcf.py --preset vsss`.
Tested with Python 3.11, MuJoCo 3.14.0, NumPy 2.4. The Gazebo part needs ROS 2 Humble, Gazebo
Fortress, `install_dependency.sh`, and `colcon build` (see `README.md`). No GPU is needed.

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

## 6. Key results for the owner's robot (preset `vsss`)

- **Top speed:** 0.76 m/s along a body axis at 11.1 V full duty. Scale by battery voltage, which
  runs 10.5–12.6 V over a charge (about ±7 %).
- **Open-loop duty feed-forward:** gives ~80 % of the commanded speed (gearbox friction, rolling
  losses over the 12 contacts, scrub between locked worms). Speed control must close the loop
  through the camera, or the feed-forward must be calibrated.
- **Traction limits** (X layout): ≈ 0.71·μ·g along the body axes and 0.5·μ·g along the diagonals.
  The motors (stall 0.023 N·m) exceed this at low speed, so full-duty starts spin the wheels a little.
- **Stops and pitch:**
  - sudden stop from full speed in brake mode: ~35 mm, ~10° pitch (~8 mm side lift);
  - 200 ms ramp: ~2 mm lift;
  - coast mode: rolls ~20 cm with ~1 mm lift.
  - The owner saw "a couple of mm" without ramps, which falls between brake and coast. Unknowns:
    driver decay mode and rotor inertia.
- **Being pushed:** the unpowered robot resists 1.70 N along a body axis and 1.32 N along a
  diagonal (predictions 1.60 / 1.13 N at μ = 1). Pushed robots skid rather than roll.
- **Vibration:** peaks around 0.4–0.5 m/s at 0.5–1.3 g rms, depending on contact softness.
- **Sim speed:** ~6–8× real time for one robot on one CPU core, too slow for large-scale MARL
  self-play (see §8).

## 7. Open items

Values to measure (procedures are in `mujoco/README.md`, "Still placeholders"):

- **Self-locking:** does a wheel turn by hand with the motor unpowered?
- **`friction`:** incline test, μ = 2·tan θ with the diagonal pointing downhill.
- **Effective radius:** distance per wheel revolution, from video.
- **`pwm_decay`:** stop distance from full speed (~4 cm brake, ~20 cm coast), or read it off the
  motor driver's datasheet and wiring (which driver is used is still unknown).
- **`wheel_armature`:** spin-up time of a lifted wheel on video.
- **`contact_timeconst` / `contact_dampratio`:** IMU vibration against speed.
- **`com_height`:** balance test.
- **Firmware ramp rates:** model them exactly in the environment.

Not done yet:

- The VSSS field (verify dimensions against current rules, commonly 150 × 130 cm with walls and
  goals), the ball (commonly an orange golf ball; verify), and multi-robot scenes. The chassis
  collision bitmask already allows robot–robot and ball contacts.
- The Gym/PettingZoo environment: action pipeline (vx, vy, ω) → ramp limiter → inverse
  kinematics with r_eff → duty feed-forward (plus deadband compensation) → latency queue →
  `Drive.step`. Observations should come from the simulated overhead tracker (no encoders on the
  real robot), with privileged state only for the critic. Also domain randomization of battery
  voltage, friction, mass/COM, motor gains and latency.
- A fast simplified robot model for MARL, e.g. planar dynamics with a wheel-level slip and traction
  model fitted to the roller model, or MJX. Keep the roller model as the reference for sysID and
  validation.
- An optional ROS 2 bridge (MuJoCo ↔ `/cmd_vel`, `/odom`, `/imu`).
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

## 9. History (branch `humble`)

- `1f6905d` MuJoCo port: generator, drive test, URDF importer.
- `2be0b4d` single-row wheels: roller length capped by neighbours.
- `d7260bf` spool/peanut rollers, effective rolling radius.
- `0f656a3` `vsss` preset for the real robot, hub collision, supply voltage.
- next commit: worm-gear drive model (`drive.py`), elliptic cones + `impratio`, launch/stop/push
  tests, COM 25 mm, contact 10 ms, this file.
