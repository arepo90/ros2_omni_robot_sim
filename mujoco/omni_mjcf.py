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
    faceting vibration); roller_shape="mesh" gives the exact profile as a mesh;
    roller_shape="peanut" gives spool / figure-8 rollers (two ellipsoid rims on one
    axle) that touch the floor with their ends, not their waist,
  * wheel i sits at angle heading_offset + 360*i/N, spin axis pointing inward
    (same sign convention as src/kinematics.cpp: +speed -> CCW tangent motion).

MuJoCo convex-hulls every mesh, so the repo's non-convex chassis/hub STLs can't be
reused as collision geometry; the chassis here is a box and the hub is visual only.

Usage:
  python omni_mjcf.py                  # writes vsss_omni4.xml with the defaults
  python omni_mjcf.py --preset vsss -o my_robot.xml   # the real robot
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
    roller_shape: str = "ellipsoid"  # "ellipsoid" (smooth, fast), "mesh" (exact barrel, faceted), "peanut" (spool: 2 rims)
    rows: int = 2  # roller rows per wheel (1 or 2)
    rollers_per_row: int = 3
    row_spacing: float = 0.006  # axial distance between the two row planes [m]
    roller_radius: float = 0.005  # max roller radius [m] (middle of a barrel, end rims of a peanut/spool)
    roller_overlap_deg: float = 4.0  # mesh rollers: extra angular coverage so rows overlap
    roller_half_length: float = 0.0  # half the roller length [m]; 0 = auto (see roller_geometry())
    roller_end_gap: float = 0.001  # min clearance between roller ends in one row (spokes/pins) [m]
    lobe_offset: float = 0.0  # peanut: roller middle -> rim centre along the axle [m]; 0 = rims evenly spaced
    lobe_half_length: float = 0.0  # peanut: rim semi-thickness along the axle [m]; 0 = 0.5 * roller_radius
    hub_mass: float = 0.008  # [kg] per wheel
    hub_radius: float = 0.0  # >0: the hub also collides with the floor as a disc this big (catches hub strikes) [m]
    hub_thickness: float = 0.0035  # [m], for the hub collision disc
    roller_mass: float = 0.0005  # [kg] per roller
    roller_damping: float = 1e-7  # bearing friction of rollers [N m s/rad]
    friction: float = 0.8  # roller/floor sliding friction
    contact_timeconst: float = 0.01  # roller contact softness [s] (>= 2*timestep); ~1/(vertical bounce freq in rad/s)
    contact_dampratio: float = 1.0  # roller contact damping ratio (rubber: < 1 is bouncier)
    # --- motor ---------------------------------------------------------------------
    motor: str = "servo"  # "servo": ideal-ish speed loop (like the Gazebo sim); "dc": voltage-driven DC motor
    servo_kv: float = 0.05  # speed-loop gain [N m / (rad/s)]
    max_wheel_speed: float = 60.0  # servo ctrl range [rad/s]
    stall_torque: float = 0.10  # at the wheel (after gearbox) [N m]; also servo torque limit
    no_load_speed: float = 62.8  # at the wheel, at nominal_voltage [rad/s]
    nominal_voltage: float = 6.0  # voltage the stall torque / no-load speed are quoted at
    supply_voltage: float = 0.0  # battery voltage the dc motor model is driven with; 0 = nominal_voltage
    wheel_armature: float = 1e-5  # reflected rotor inertia J_rotor * gear^2 [kg m^2]
    # --- simulation ------------------------------------------------------------
    timestep: float = 0.0005


PRESETS = {
    # placeholder 75 mm robot with a double-row wheel
    "generic": {},
    # the real robot: 4x GA12-N20 12 V 381 rpm (90-degree output), 6 silicone spool rollers per wheel
    "vsss": dict(
        n_wheels=4, heading_offset_deg=-45.0, wheel_R=0.034,  # wheels on the diagonals, 34 mm out
        body_size=0.075, body_mass=0.214,  # 230 g total minus ~4 g per wheel
        com_height=0.020,  # PLACEHOLDER
        wheel_radius=0.01725, roller_shape="peanut", rows=1, rollers_per_row=6,  # wheel fits a 34.5 mm circle
        roller_radius=0.00335, lobe_half_length=0.00128,  # 6.7 mm ends; rims sized for a 9.8 mm roller
        hub_radius=0.0157, hub_thickness=0.0034,  # hub fits a 31.4 mm circle
        hub_mass=0.0022, roller_mass=0.0003,  # estimated from PLA / silicone volumes
        friction=1.0,  # PLACEHOLDER: silicone on the field surface
        contact_timeconst=0.005,  # Hertz estimate for ~A35 silicone rims: ~0.35 mm squash, ~33 Hz bounce
        nominal_voltage=12.0, no_load_speed=39.9, max_wheel_speed=39.9,  # 381 rpm at 12 V
        stall_torque=0.06,  # PLACEHOLDER [N m]: from the listing / a stall test
        wheel_armature=2e-5,  # PLACEHOLDER: N20 rotor inertia x gear ratio^2
    ),
}


def make_params(preset="generic", **overrides) -> OmniParams:
    return OmniParams(**{**PRESETS[preset], **overrides})


def add_param_args(ap: argparse.ArgumentParser):
    """--preset plus one --<field> option per OmniParams field, defaulting to the preset's value."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    base = make_params(pre.parse_known_args()[0].preset)
    ap.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    for f in fields(OmniParams):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=getattr(base, f.name))


def params_from_args(a: dict) -> OmniParams:
    a.pop("preset", None)
    return OmniParams(**{k: v for k, v in a.items() if k in {f.name for f in fields(OmniParams)}})


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


@dataclass
class RollerGeometry:
    d: float  # hub centre -> roller axis
    half_len: float  # roller semi-length (peanut: semi-length of each lobe)
    lobe_offset: float  # peanut: roller middle -> lobe centre along the axle (0 otherwise)
    max_half_len: float  # longest half_len that keeps roller_end_gap to the same-row neighbour
    env_min: float  # rolling radius range over a revolution (the wheel's roundness)
    env_max: float
    r_eff: float  # effective rolling radius = distance per revolution / 2pi (use it in the kinematics)


def _outline(p: OmniParams, half_len, s_l=0.0, n=61):
    """Roller cross-section in the wheel plane as (radial, axial) points around its centre."""
    rho0 = p.roller_radius
    if p.roller_shape == "mesh":
        r, d = p.wheel_radius, p.wheel_radius - rho0
        s = np.linspace(-half_len, half_len, n)
        rho = np.sqrt(r * r - s * s) - d
        cap = np.linspace(-rho[0], rho[0], 11)
        return np.concatenate([
            np.stack([rho, s], 1), np.stack([-rho, s], 1),
            np.stack([cap, np.full_like(cap, s[0])], 1), np.stack([cap, np.full_like(cap, s[-1])], 1),
        ])
    t = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    ell = np.stack([rho0 * np.cos(t), half_len * np.sin(t)], 1)
    return ell if p.roller_shape == "ellipsoid" else np.concatenate([ell + [0, s_l], ell - [0, s_l]])


def _place(outline, d, phi):
    """Outline of a roller whose axis is at distance d, centred at wheel angle phi."""
    er, et = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
    return d * er + outline[:, :1] * er + outline[:, 1:] * et


def _clear(p: OmniParams, d, half_len, s_l=0.0):
    """Do neighbouring rollers of one row keep roller_end_gap between them?"""
    o = _outline(p, half_len, s_l)
    A, B = _place(o, d, 0.0), _place(o, d, 2 * math.pi / p.rollers_per_row)
    return np.min(np.linalg.norm(A[:, None] - B[None], axis=2)) >= max(p.roller_end_gap, 1e-5)


def _largest(ok, lo, hi, iters=30):
    """Largest x in [lo, hi] with ok(x), for ok true below some threshold."""
    if ok(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _envelope(p: OmniParams, d, half_len, s_l=0.0, n=1441):
    o = _outline(p, half_len, s_l)
    pts = np.concatenate([
        _place(o, d, 2 * np.pi * j / p.rollers_per_row + k * np.pi / p.rollers_per_row)
        for k in range(p.rows) for j in range(p.rollers_per_row)
    ])
    psi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (pts @ np.stack([np.cos(psi), np.sin(psi)])).max(0)


def roller_geometry(p: OmniParams) -> RollerGeometry:
    """Where the rollers sit and how long they are.

    Ellipsoid: sqrt(rho0 * r) matches the wheel curvature at the roller middle; a slightly
      longer roller flattens the dips where contact hands over, so the length is tuned for
      roundness, capped so rollers in one row keep roller_end_gap. roller_half_length overrides.
    Mesh: the exact barrel profile over 180/(rows*n) + roller_overlap_deg, same cap.
    Peanut / spool: the contact rims at the two roller ends, each an ellipsoid of radius
      roller_radius and axial semi-length lobe_half_length (default 0.5 * roller_radius: a
      disc with a rounded edge), with the rim tips on the rolling circle. By default the
      2*n rims of a row are evenly spaced around the wheel (6 rollers -> a contact every 30
      deg); lobe_offset places them explicitly (roller middle -> rim centre along the axle).
    """
    r, rho = p.wheel_radius, p.roller_radius
    c = r - rho
    if p.roller_shape == "peanut":
        beta0 = math.pi / (2 * p.rows * p.rollers_per_row)

        def place(a):
            """(lobe offset, roller axis distance) with the rim tips on the rolling circle."""
            tip = lambda b: math.sqrt((rho * math.cos(b)) ** 2 + (a * math.sin(b)) ** 2)
            if p.lobe_offset <= 0:
                cc = r - tip(beta0)
                return cc * math.sin(beta0), cc * math.cos(beta0)
            b = math.asin(min(p.lobe_offset / c, 0.99))
            for _ in range(20):
                cc = r - tip(b)
                b = math.asin(min(p.lobe_offset / cc, 0.99))
            if p.lobe_offset >= cc:
                raise ValueError(f"lobe_offset {p.lobe_offset*1e3:.2f} mm is too large for this wheel")
            return p.lobe_offset, math.sqrt(cc * cc - p.lobe_offset ** 2)

        ok = lambda x: _clear(p, place(x)[1], x, place(x)[0])
        if not ok(1e-4):
            raise ValueError("rims of neighbouring rollers collide: reduce roller_radius, "
                             "roller_end_gap or lobe_offset")
        a_max = _largest(ok, 1e-4, 2 * rho)
        a = p.lobe_half_length or min(0.5 * rho, a_max)
        s_l, d = place(a)
    else:
        s_l, d = 0.0, c
        hi = 0.98 * math.sqrt(r * r - d * d) if p.roller_shape == "mesh" else 3 * math.sqrt(rho * r)
        a_max = _largest(lambda x: _clear(p, d, x), 1e-5, hi)
        if p.roller_half_length > 0:
            a = p.roller_half_length
        elif p.roller_shape == "mesh":
            alpha = math.radians(180.0 / (p.rows * p.rollers_per_row) + p.roller_overlap_deg)
            a = min(r * math.sin(alpha), a_max)
        else:
            a0 = math.sqrt(rho * r)
            hi = min(1.6 * a0, a_max)
            a = min(np.linspace(min(a0, hi), hi, 61), key=lambda x: np.ptp(_envelope(p, d, x)))
    env = _envelope(p, d, a, s_l)
    # rolling without slip on a convex profile covers its hull perimeter = integral of the
    # support function, so distance per revolution / 2pi is simply the mean rolling radius
    return RollerGeometry(d, a, s_l, a_max, env.min(), env.max(), env.mean())


def describe_rollers(p: OmniParams, g: RollerGeometry) -> str:
    txt = f"{p.rows}x{p.rollers_per_row} {p.roller_shape} rollers: "
    if p.roller_shape == "peanut":
        txt += (f"rims r={p.roller_radius*1e3:.2f} mm, {2*g.half_len*1e3:.2f} mm thick, at +-{g.lobe_offset*1e3:.2f} mm "
                f"(+-{math.degrees(math.atan2(g.lobe_offset, g.d)):.1f} deg), roller length "
                f"{2*(g.lobe_offset+g.half_len)*1e3:.1f} mm, axle {g.d*1e3:.2f} mm from hub")
    else:
        txt += f"half-length {g.half_len*1e3:.2f} mm (same-row limit {g.max_half_len*1e3:.2f} mm)"
    return txt + (f"; rolling radius {g.env_min*1e3:.3f}..{g.env_max*1e3:.3f} mm "
                  f"(bump {(g.env_max-g.env_min)*1e3:.3f} mm), effective {g.r_eff*1e3:.3f} mm")


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
    if p.roller_shape not in ("ellipsoid", "mesh", "peanut"):
        raise ValueError("roller_shape must be 'ellipsoid', 'mesh' or 'peanut'")
    if p.rows not in (1, 2):
        raise ValueError("rows must be 1 or 2")
    r = p.wheel_radius
    g = roller_geometry(p)
    d, a, a_max = g.d, g.half_len, g.max_half_len  # d: hub centre -> roller axis
    if a > a_max * 1.001:
        what = "lobe_half_length" if p.roller_shape == "peanut" else "roller_half_length"
        warnings.warn(f"{what} {a*1e3:.2f} mm overlaps the neighbouring roller "
                      f"(max {a_max*1e3:.2f} mm with roller_end_gap {p.roller_end_gap*1e3:g} mm)")
    roller = 'contype="2" conaffinity="1"'
    ell = (f'type="sphere" size="{p.roller_radius:g}"' if abs(a - p.roller_radius) < 1e-9 else
           f'type="ellipsoid" size="{p.roller_radius:g} {p.roller_radius:g} {a:.5g}"')
    mesh_asset = ""
    if p.roller_shape == "mesh":
        roller_geoms = f'<geom class="roller" type="mesh" mesh="roller" mass="{p.roller_mass:g}" {roller}/>'
        mesh_asset = f'<mesh name="roller" vertex="{_fmt(_roller_mesh(p, a))}"/>'
    elif p.roller_shape == "peanut":  # non-convex, so two convex lobes (a mesh would be hulled into a barrel)
        roller_geoms = "".join(
            f'<geom class="roller" {ell} pos="0 0 {z:.5g}" mass="{p.roller_mass/2:g}" {roller}/>'
            for z in (-g.lobe_offset, g.lobe_offset))
    else:
        roller_geoms = f'<geom class="roller" {ell} mass="{p.roller_mass:g}" {roller}/>'
    # Contact bits: floor=1; rollers collide only with the floor/ball (contype 2);
    # chassis (contype 4) collides with floor and other chassis, never own rollers.
    floor = 'contype="1" conaffinity="1"'
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

    if p.hub_radius > 0:  # PLA hub touching the floor when the rubber rollers squash
        hub = (f'<geom type="cylinder" size="{p.hub_radius:g} {p.hub_thickness/2:g}" mass="{p.hub_mass:g}" '
               f'priority="1" friction="0.3 0.005 0.0001" solref="0.002 1" rgba=".1 .1 .1 1" {roller}/>')
    else:
        hub = (f'<geom type="cylinder" size="{d*0.9:.5g} {p.row_spacing/2 + p.roller_radius*0.6:.5g}" '
               f'mass="{p.hub_mass:g}" rgba=".8 .8 .8 1" {visual}/>')

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
                    + roller_geoms +
                    "</body>"
                )
        wheels.append(
            f'<body name="wheel_{i}" pos="{_fmt(pos)}" zaxis="{_fmt(axle)}">'
            f'<joint name="wheel_{i}" axis="0 0 1" armature="{p.wheel_armature:g}"/>'
            + hub
            + "".join(rollers)
            + "</body>"
        )

    volts = p.supply_voltage or p.nominal_voltage
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
                f'ctrlrange="{-volts:g} {volts:g}"/>'
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
      <geom priority="1" friction="{p.friction:g} 0.005 0.0001" solref="{p.contact_timeconst:g} {p.contact_dampratio:g}" rgba=".1 .1 .1 1"/>
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
    add_param_args(ap)
    a = vars(ap.parse_args())
    out = a.pop("out")
    p = params_from_args(a)
    with open(out, "w") as fh:
        fh.write(build_mjcf(p))
    print(f"wrote {out}")
    g = roller_geometry(p)
    print(describe_rollers(p, g))
    print(f"use r = {g.r_eff*1e3:.3f} mm (effective rolling radius) in the kinematics")
    print("wheel jacobian (rad/s per [vx, vy, wz]):\n", np.round(wheel_jacobian(p), 3))


if __name__ == "__main__":
    main()
