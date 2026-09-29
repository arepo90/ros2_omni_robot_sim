#!/usr/bin/env python3
"""IEEE VSSS field with the ball, robots and an overhead camera, as one MuJoCo model.

  python3 vsss_field.py                      # writes vsss_field.xml: 3 blue robots, the ball, the camera
  python3 vsss_field.py --blue 3 --yellow 3  # both teams
  python3 vsss_field.py --model roller       # the detailed roller-level robots instead of the fast ones
  python3 -m mujoco.viewer --mjcf=vsss_field.xml

The viewer only shows the scene. To drive the robots, step it with planar_robot.PlanarDrive
(model "planar", the default) or drive.Drive (model "roller", one per robot, prefix "blue_0/"),
as ros_bridge.py does.

Field: 3v3 (FIRASim "Division B" defaults, the league's simulator): 150 x 130 cm matte black
floor, 5 cm walls 2.5 cm thick, 40 x 10 cm goals, 3 mm white lines, 20 cm centre circle,
70 x 15 cm defense areas, free-ball marks where FIRASim draws them, and 7 cm isosceles
triangles in the corners. Ball: orange golf ball, 42.7 mm, 46 g. Check against the current rules.

Frame "field": origin at the centre, x towards the yellow goal, y to the left, z up. Blue defends
-x. The overhead camera looks straight down from `cam_z`; in its image +x is right and +y is up.

Robots are the `vsss` preset (omni_mjcf.py): a black 7.5 cm cube with the team's 3-colour top
(`_add_top`): the team colour across the front half, two ID colours side by side on the rear half
(left, right as seen from above with the front up), 4 mm black borders. Colours and ID pairs are
sampled from the team's pattern sheet. The ball and the walls are not validated physics.
"""
import argparse
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from omni_mjcf import OmniParams, build_mjcf, make_params
from planar_robot import PlanarParams, robot_spec as planar_robot_spec

FIELD_L, FIELD_W = 1.50, 1.30  # playing area inside the walls [m]
WALL_T, WALL_H = 0.025, 0.05
GOAL_W, GOAL_D = 0.40, 0.10
CORNER_LEG = 0.07  # corner triangles
LINE_W = 0.003
CENTER_R = 0.20  # outer radius of the centre circle
AREA_W, AREA_D = 0.70, 0.15  # defense area (FIRASim "penalty width / depth")
PENALTY_X = 0.375  # penalty marks at (+-0.375, 0)
FREE_BALL = (FIELD_L / 4, FIELD_W / 2 * 2 / 3)  # free-ball marks at (+-x, +-y), as FIRASim draws them
BALL_R, BALL_MASS = 0.02135, 0.046

# robot tops, from the team's pattern sheet (sRGB 0-255)
_rgb = lambda r, g, b: [r / 255, g / 255, b / 255, 1]
TEAM_RGBA = {"blue": _rgb(5, 11, 159), "yellow": _rgb(255, 230, 13)}
RED, GREEN, CYAN, MAGENTA = _rgb(204, 0, 1), _rgb(0, 204, 8), _rgb(0, 170, 206), _rgb(205, 23, 220)
ID_PAIRS = [(RED, GREEN), (RED, CYAN), (GREEN, RED), (GREEN, CYAN), (GREEN, MAGENTA),  # (left, right), robot 0..9
            (CYAN, RED), (CYAN, GREEN), (CYAN, MAGENTA), (MAGENTA, GREEN), (MAGENTA, CYAN)]
TOP_BORDER, TOP_GAP = 0.004, 0.004  # black rim around the patches and between them
CHASSIS_RGBA = [0.05, 0.05, 0.05, 1]
# start poses (x, y, yaw) for blue; yellow is mirrored
BLUE_SLOTS = [(-0.30, 0.0), (-0.45, 0.30), (-0.65, 0.0), (-0.45, -0.30), (-0.20, -0.40)]

FLOOR_RGBA = [0.03, 0.03, 0.03, 1]  # matte black
WALL_RGBA = [0.06, 0.06, 0.06, 1]
LINE_RGBA = [1, 1, 1, 1]
BALL_RGBA = [1.0, 0.45, 0.0, 1]
STATIC = dict(contype=1, conaffinity=1)  # like the floor: collides with robots (chassis, rollers) and the ball
VISUAL = dict(contype=0, conaffinity=0, group=1)


@dataclass
class Robot:
    name: str  # also the MuJoCo name prefix (+ "/") and the ROS namespace
    team: str
    index: int
    start: tuple  # x, y, yaw


@dataclass
class Scene:
    spec: mujoco.MjSpec
    p: OmniParams
    model: str  # "planar" or "roller"
    q: PlanarParams
    robots: list = field(default_factory=list)
    cam: dict = field(default_factory=dict)  # name, z, fovy, width, height, ortho


def _box(body, name, pos, half, rgba, yaw=0.0, **kw):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba, material="matte",
                  quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], **kw)


def _add_field(spec: mujoco.MjSpec):
    wb = spec.worldbody
    hx, hy = FIELD_L / 2, FIELD_W / 2
    # physical floor (infinite plane), drawn only under the field; a grey visual slab around it
    spec.add_material(name="matte", specular=0, shininess=0, reflectance=0)  # painted MDF, no highlights
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[hx + GOAL_D + WALL_T, hy + WALL_T, 0.05],
                material="matte", rgba=FLOOR_RGBA, **STATIC)
    _box(wb, "surround", [0, 0, -0.002], [3, 3, 0.001], [0.35, 0.35, 0.37, 1], **VISUAL)

    # walls: sides along x, ends split by the goal mouth, goal boxes, corner triangles
    zc, hh = WALL_H / 2, WALL_H / 2
    for s in (1, -1):
        _box(wb, f"wall_side_{'l' if s > 0 else 'r'}", [0, s * (hy + WALL_T / 2), zc], [hx + WALL_T, WALL_T / 2, hh],
             WALL_RGBA, **STATIC)
        seg = (hy - GOAL_W / 2) / 2  # half length of an end-wall segment
        for e in (1, -1):
            _box(wb, f"wall_end_{s}_{e}", [s * (hx + WALL_T / 2), e * (GOAL_W / 2 + seg), zc], [WALL_T / 2, seg, hh],
                 WALL_RGBA, **STATIC)
            _box(wb, f"goal_side_{s}_{e}", [s * (hx + GOAL_D / 2), e * (GOAL_W / 2 + WALL_T / 2), zc],
                 [GOAL_D / 2 + WALL_T / 2, WALL_T / 2, hh], WALL_RGBA, **STATIC)
            # triangular prism in the corner (convex, so its collision shape is exact)
            cx, cy, L = s * hx, e * hy, CORNER_LEG
            tri = [(cx, cy), (cx - s * L, cy), (cx, cy - e * L)]
            spec.add_mesh(name=f"corner_{s}_{e}", uservert=[v for x, y in tri for z in (0, WALL_H) for v in (x, y, z)])
            wb.add_geom(name=f"corner_{s}_{e}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"corner_{s}_{e}",
                        material="matte", rgba=WALL_RGBA, **STATIC)
        _box(wb, f"goal_back_{s}", [s * (hx + GOAL_D + WALL_T / 2), 0, zc], [WALL_T / 2, GOAL_W / 2 + WALL_T, hh],
             WALL_RGBA, **STATIC)

    # white lines: thin visual slabs just above the floor
    z, t = 0.00025, 0.00025
    lw = LINE_W / 2

    def line(name, x0, y0, x1, y1):
        cx, cy, L = (x0 + x1) / 2, (y0 + y1) / 2, math.hypot(x1 - x0, y1 - y0)
        _box(wb, name, [cx, cy, z], [L / 2 + lw, lw, t], LINE_RGBA, yaw=math.atan2(y1 - y0, x1 - x0), **VISUAL)

    bx, by = hx - lw, hy - lw  # border lines run along the inside of the walls
    line("line_top", -bx, by, bx, by)
    line("line_bottom", -bx, -by, bx, -by)
    line("line_mid", 0, -by, 0, by)
    for s in (1, -1):
        line(f"line_end_{s}", s * bx, -by, s * bx, by)
        ax = s * (hx - AREA_D + lw)
        line(f"area_front_{s}", ax, -AREA_W / 2, ax, AREA_W / 2)
        for e in (1, -1):
            line(f"area_side_{s}_{e}", s * bx, e * (AREA_W / 2 - lw), ax, e * (AREA_W / 2 - lw))
    n = 64
    rc = CENTER_R - lw
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        line(f"circle_{k}", rc * math.cos(a0), rc * math.sin(a0), rc * math.cos(a1), rc * math.sin(a1))
    marks = [(s * PENALTY_X, 0) for s in (1, -1)] + [(s * FREE_BALL[0], e * FREE_BALL[1]) for s in (1, -1) for e in (1, -1)]
    for k, (x, y) in enumerate(marks):
        wb.add_geom(name=f"mark_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, z], size=[0.004, t, 0],
                    rgba=LINE_RGBA, **VISUAL)


def _add_ball(spec: mujoco.MjSpec):
    ball = spec.worldbody.add_body(name="ball", pos=[0, 0, BALL_R + 0.001])
    ball.add_freejoint(name="ball")
    # condim 6: rolling resistance (a guess: ~0.2 m/s^2 of deceleration on MDF), not measured
    ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_MASS,
                  rgba=BALL_RGBA, condim=6, friction=[0.8, 0.005, 0.0006], **STATIC)


def _add_top(body, spec: mujoco.MjSpec, p: OmniParams, team: str, index: int, z_top: float):
    """The 3-colour top: team colour on the front half (+x), ID colours on the rear half, left (+y)
    and right (-y) seen from above with the front up."""
    spec.add_material(name="matte", specular=0, shininess=0, reflectance=0)
    inner = p.body_size - 2 * TOP_BORDER
    depth = (inner - TOP_GAP) / 2  # each half, front to back
    xc = TOP_GAP / 2 + depth / 2
    left, right = ID_PAIRS[index % len(ID_PAIRS)]
    for name, pos, half, rgba in (("team_patch", [xc, 0], [depth / 2, inner / 2], TEAM_RGBA[team]),
                                  ("id_left", [-xc, xc], [depth / 2, depth / 2], left),
                                  ("id_right", [-xc, -xc], [depth / 2, depth / 2], right)):
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*pos, z_top + 0.0005], size=[*half, 0.0005],
                      rgba=rgba, material="matte", mass=0, **VISUAL)


def _robot_spec(p: OmniParams, team: str, index: int, model: str, q: PlanarParams) -> mujoco.MjSpec:
    if model == "planar":
        rs = planar_robot_spec(p, q, rgba=CHASSIS_RGBA)
        _add_top(rs.body("chassis"), rs, p, team, index, p.body_height)  # chassis frame on the floor
        return rs
    rs = mujoco.MjSpec.from_string(build_mjcf(p))
    chassis = rs.body("chassis")
    for g in list(chassis.geoms):
        if g.name == "chassis":
            g.rgba = CHASSIS_RGBA
        elif g.type == mujoco.mjtGeom.mjGEOM_BOX:  # the red forward marker; the top pattern marks the front now
            rs.delete(g)
    _add_top(chassis, rs, p, team, index, p.body_height - p.wheel_radius)  # chassis frame at axle height
    return rs


def _add_camera(spec: mujoco.MjSpec, z, width, height, fovy, ortho):
    """Straight-down camera over the centre. fovy 0 = fit the field, goals and walls with a 5 cm margin."""
    half_x = FIELD_L / 2 + GOAL_D + WALL_T + 0.05
    half_y = FIELD_W / 2 + WALL_T + 0.05
    need = max(half_y, half_x * height / width)  # half the image height needed on the floor [m]
    if not fovy:
        fovy = 2 * need if ortho else math.degrees(2 * math.atan(need / z))
    spec.worldbody.add_camera(name="overhead", pos=[0, 0, z], quat=[1, 0, 0, 0], fovy=fovy,
                              resolution=[width, height],
                              proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC if ortho else mujoco.mjtProjection.mjPROJ_PERSPECTIVE)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    return dict(name="overhead", z=z, fovy=fovy, width=width, height=height, ortho=ortho)


def build_scene(blue=3, yellow=0, p: OmniParams = None, model="planar", q: PlanarParams = None, cam_z=2.0,
                cam_width=640, cam_height=480, cam_fovy=0.0, cam_ortho=False) -> Scene:
    p = p or make_params("vsss")
    q = q or PlanarParams()
    if blue > len(BLUE_SLOTS) or yellow > len(BLUE_SLOTS):
        raise ValueError(f"at most {len(BLUE_SLOTS)} robots per team")
    if model not in ("planar", "roller"):
        raise ValueError("model must be 'planar' or 'roller'")
    spec = mujoco.MjSpec()
    spec.modelname = "vsss_field"
    if model == "roller":
        robot0 = mujoco.MjSpec.from_string(build_mjcf(p))
        for attr in ("timestep", "integrator", "cone", "impratio"):  # the roller model's validated solver settings
            setattr(spec.option, attr, getattr(robot0.option, attr))
    else:  # explicit Euler: see planar_robot.py
        spec.option.timestep, spec.option.integrator = q.timestep, mujoco.mjtIntegrator.mjINT_EULER
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # lighting adds up to 1 on surfaces facing the overhead camera, so the tops render in their own colours
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.worldbody.add_light(name="field_light", pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.3, 0.3, 0.3],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)  # even, like field lighting
    _add_field(spec)
    _add_ball(spec)
    scene = Scene(spec, p, model, q)
    for team, n in (("blue", blue), ("yellow", yellow)):
        for i in range(n):
            x, y = BLUE_SLOTS[i]
            x, y, yaw = (x, y, 0.0) if team == "blue" else (-x, -y, math.pi)
            name = f"{team}_{i}"
            frame = spec.worldbody.add_frame(pos=[x, y, 0], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            frame.attach_body(_robot_spec(p, team, i, model, q).body("chassis"), f"{name}/", "")
            scene.robots.append(Robot(name, team, i, (x, y, yaw)))
    scene.cam = _add_camera(spec, cam_z, cam_width, cam_height, cam_fovy, cam_ortho)
    return scene


def add_scene_args(ap: argparse.ArgumentParser):
    ap.add_argument("--blue", type=int, default=3, help="blue robots (0-5)")
    ap.add_argument("--yellow", type=int, default=0, help="yellow robots (0-5)")
    ap.add_argument("--model", choices=("planar", "roller"), default="planar",
                    help="planar: fast box + wheel forces (default); roller: detailed roller-level wheels")
    ap.add_argument("--cam_z", type=float, default=2.0, help="overhead camera height [m]")
    ap.add_argument("--cam_width", type=int, default=640)
    ap.add_argument("--cam_height", type=int, default=480)
    ap.add_argument("--cam_fovy", type=float, default=0.0,
                    help="vertical field of view [deg] (orthographic: image height [m]); 0 = fit the field")
    ap.add_argument("--cam_ortho", action="store_true", help="orthographic camera: flat map, no parallax")


def scene_from_args(a) -> Scene:
    return build_scene(a.blue, a.yellow, model=a.model, cam_z=a.cam_z, cam_width=a.cam_width, cam_height=a.cam_height,
                       cam_fovy=a.cam_fovy, cam_ortho=a.cam_ortho)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_scene_args(ap)
    ap.add_argument("-o", "--out", default="vsss_field.xml")
    a = ap.parse_args()
    scene = scene_from_args(a)
    m = scene.spec.compile()
    with open(a.out, "w") as fh:
        fh.write(scene.spec.to_xml())
    print(f"wrote {a.out}: {len(scene.robots)} robots ({', '.join(r.name for r in scene.robots)}), ball, "
          f"camera {scene.cam['width']}x{scene.cam['height']} at {scene.cam['z']} m, fovy {scene.cam['fovy']:.1f}"
          f"{' m (orthographic)' if scene.cam['ortho'] else ' deg'}; {m.nbody - 1} bodies, {m.ngeom} geoms")


if __name__ == "__main__":
    main()
