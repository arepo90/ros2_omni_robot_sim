#!/usr/bin/env python3
"""Calibrate a real overhead camera for vision_node.py: click field landmarks, get the camera pose.

  python3 vision/calibrate.py --device 2 --camera_height 1.95 -o vision/calib/real.yaml
  python3 vision/calibrate.py --image /camera/image_raw --camera_info /camera/camera_info -o vision/calib/real.yaml
  python3 vision/vision_node.py --calib vision/calib/real.yaml --image /camera/image_raw

Click the 6 landmarks in the order shown (white line intersections, which stay visible, unlike the
field corners under the corner triangles). With the yellow goal on the right of the image:
  1-2 left defense area, front corners: top, bottom   3-4 right defense area, front corners: top, bottom
  5-6 halfway line: top end, bottom end
Then, optionally (--colors), click one patch of each colour so the vision matches this camera's
colours instead of the printed ones: team blue, team yellow, red, green, light blue, pink, ball.

Intrinsics: from --camera_info (a checkerboard calibration, e.g. ros2 run camera_calibration
cameracalibrator) if you have one. Otherwise from --fovy, and then --camera_height (lens to floor,
measured) fixes the focal length: from straight above, a longer lens looks the same as a higher
camera, so the floor alone can't tell them apart, and the height matters for the parallax
correction of the robot tops. Lens distortion is assumed small without --camera_info.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import CameraModel  # noqa: E402

# field landmarks (x, y, 0) [m], vision/patterns.yaml / mujoco/vsss_field.py geometry
LANDMARKS = [("left defense area, front-top corner", (-0.60, 0.35)),
             ("left defense area, front-bottom corner", (-0.60, -0.35)),
             ("right defense area, front-top corner", (0.60, 0.35)),
             ("right defense area, front-bottom corner", (0.60, -0.35)),
             ("halfway line, top end", (0.0, 0.65)),
             ("halfway line, bottom end", (0.0, -0.65))]
COLOR_ORDER = ["darkblue", "yellow", "red", "green", "lightblue", "pink", "ball"]


def solve(uv, K, dist, size, camera_height=None):
    """Camera model from clicked pixels of LANDMARKS. With camera_height, rescale the focal length
    until the solved camera sits at that height."""
    obj = np.array([[x, y, 0.0] for _, (x, y) in LANDMARKS])
    img = np.asarray(uv, dtype=float)
    K = np.array(K, dtype=float)
    for i in range(20):
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)  # IPPE flips to a bad pose from straight above
        if not ok:
            raise RuntimeError("solvePnP failed: check the click order")
        R, _ = cv2.Rodrigues(rvec)
        z = float((-R.T @ tvec.ravel())[2])
        if camera_height is None or abs(z / camera_height - 1) < 1e-5 or i == 19:
            break
        K[0, 0] *= camera_height / z  # from straight above, apparent size ~ f / height
        K[1, 1] *= camera_height / z
    cam = CameraModel(K, dist, R, tvec.ravel(), size)
    err = np.linalg.norm(cam.to_pixel(obj) - img, axis=1)
    return cam, err


def overlay(img, cam):
    """Draw the field lines where the calibration says they are."""
    out = img.copy()
    hx, hy = 0.75, 0.65
    lines = [((-hx, hy), (hx, hy)), ((hx, hy), (hx, -hy)), ((hx, -hy), (-hx, -hy)), ((-hx, -hy), (-hx, hy)),
             ((0, hy), (0, -hy)), ((-0.6, 0.35), (-0.6, -0.35)), ((0.6, 0.35), (0.6, -0.35))]
    for a, b in lines:
        pts = cam.to_pixel(np.array([[*a, 0.0], [*b, 0.0]])).astype(int)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 1)
    circle = np.array([[0.2 * math.cos(t), 0.2 * math.sin(t), 0.0] for t in np.linspace(0, 2 * math.pi, 64)])
    cv2.polylines(out, [cam.to_pixel(circle).astype(np.int32)], True, (0, 0, 255), 1)
    return out


def click(img, prompts):
    """Show img, collect one click per prompt; Esc aborts."""
    pts, win = [], "calibrate"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: pts.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while len(pts) < len(prompts):
        show = img.copy()
        for p in pts:
            cv2.circle(show, p, 4, (0, 0, 255), -1)
        cv2.putText(show, f"{len(pts) + 1}/{len(prompts)}: {prompts[len(pts)]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, show)
        if cv2.waitKey(20) == 27:
            raise SystemExit("aborted")
    cv2.destroyWindow(win)
    return pts


def grab(a):
    """One BGR frame (and CameraInfo, if asked) from a device or a ROS topic."""
    if a.device is not None:
        cap = cv2.VideoCapture(a.device)
        for _ in range(10):  # let exposure settle
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"no frame from camera {a.device}")
        return frame, None
    import rclpy
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node, got = rclpy.create_node("calibrate"), {}
    node.create_subscription(Image, a.image, lambda m: got.setdefault("img", m), 1)
    if a.camera_info:
        node.create_subscription(CameraInfo, a.camera_info, lambda m: got.setdefault("info", m), 1)
    while "img" not in got or (a.camera_info and "info" not in got):
        rclpy.spin_once(node, timeout_sec=0.5)
    m = got["img"]
    img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else img.copy()
    rclpy.shutdown()
    return frame, got.get("info")


def main():
    ap = argparse.ArgumentParser(description="Click field landmarks -> camera calibration YAML (see the docstring)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--device", type=int, help="cv2.VideoCapture index of the camera")
    src.add_argument("--image", help="ROS image topic")
    ap.add_argument("--camera_info", help="ROS CameraInfo topic with calibrated intrinsics")
    ap.add_argument("--fovy", type=float, default=60.0, help="vertical field of view guess [deg] without camera_info")
    ap.add_argument("--camera_height", type=float, help="lens to floor [m]: fixes the focal length without camera_info")
    ap.add_argument("--colors", action="store_true", help="also click one patch of each colour")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent / "calib" / "real.yaml"))
    a = ap.parse_args()
    frame, info = grab(a)
    h, w = frame.shape[:2]
    if info is not None:
        K, dist, height = np.array(info.k).reshape(3, 3), np.array(info.d if len(info.d) else [0.0] * 5), None
    else:
        f = (h / 2) / math.tan(math.radians(a.fovy) / 2)
        K, dist, height = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]), np.zeros(5), a.camera_height
        if height is None:
            print("warning: no --camera_info or --camera_height; the focal length is a guess, so the robot-top "
                  "parallax correction will be off")
    uv = click(frame, [name for name, _ in LANDMARKS])
    cam, err = solve(uv, K, dist, (w, h), height)
    print(f"camera at {np.round(cam.center, 3).tolist()} m, focal {cam.K[0, 0]:.0f} px, "
          f"click reprojection error {err.mean():.1f} px (max {err.max():.1f})")
    extra = {}
    if a.colors:
        pts = click(frame, [f"a {c} patch" for c in COLOR_ORDER])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        extra["colors"] = {c: [int(v) for v in np.median(rgb[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3), 0)]
                           for c, (x, y) in zip(COLOR_ORDER, pts)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cam.save(a.out, extra)
    print(f"wrote {a.out}; check the red field lines, then run vision_node.py --calib {a.out}")
    cv2.imshow("calibration check (any key closes)", overlay(frame, cam))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
