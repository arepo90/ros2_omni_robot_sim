#!/usr/bin/env python3
"""Vision node: overhead camera image -> global field state (/field, vsss_msgs/Field).

  source install/setup.bash                # vsss_msgs (colcon build --base-paths msgs)
  python3 vision/vision_node.py            # sim: camera pose from TF + camera_info
  python3 vision/vision_node.py --calib vision/calib/real.yaml --image /camera/image_raw   # a real camera

The same pipeline as the old YOLO vision (vsss repo, commit 7490988), split in two loops so a 60 Hz
camera never waits for the detector (12 ms on the RTX 4060, ~40 runs/s next to the frame loop):
  * YOLO (vision/models/robots_yolov8m.pt, class "robot") runs in a background thread on the latest
    frame (up to --yolo_hz, 10/s) and finds robots;
  * every frame, the colour patches are decoded (vision/detect.py) in a window around each tracked
    robot (predicted to that frame) and around each YOLO box no track accounts for: team, id,
    position and heading on the plane of the robot tops; the ball is found anywhere;
  * a Kalman filter per robot id and one for the ball (vision/track.py) smooth them and give
    velocities. /field is stamped with the camera frame's time.

Topics: in  <--image> (sensor_msgs/Image, rgb8 or bgr8), <--camera_info>, TF field -> <camera frame>
        out /field (vsss_msgs/Field), /vision/image_annotated (only while someone subscribes)
"""
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vsss_msgs.msg import Field, Object

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import CameraModel  # noqa: E402
from detect import ColorLUT, Decoder, YoloRobots, load_patterns  # noqa: E402
from track import Tracker, TrackParams  # noqa: E402

HERE = Path(__file__).resolve().parent


class YoloThread:
    """Runs the detector on the newest frame it's given; keeps the latest result."""

    def __init__(self, detector, max_hz):
        self.detector, self.period = detector, 1.0 / max_hz
        self.frame, self.result = None, (None, np.zeros((0, 4)), np.zeros(0))  # (stamp, boxes, confs)
        self.cv = threading.Condition()
        self.runs, self.busy, self.stopping = 0, False, False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Let the thread finish before Python exits: a daemon thread inside CUDA hangs the exit."""
        with self.cv:
            self.stopping = True
            self.cv.notify()
        self.thread.join(timeout=5)

    def offer(self, stamp, rgb):
        with self.cv:
            if not self.busy:
                self.frame = (stamp, rgb)
                self.cv.notify()

    def _loop(self):
        while True:
            with self.cv:
                while self.frame is None and not self.stopping:
                    self.cv.wait()
                if self.stopping:
                    return
                (stamp, rgb), self.frame, self.busy = self.frame, None, True
            t0 = time.monotonic()
            boxes, confs = self.detector(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # its Python side holds the GIL; tracks cover known robots, so leave time for the frames
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))
            with self.cv:
                self.result, self.busy = (stamp, boxes, confs), False
                self.runs += 1


class Vision(Node):
    def __init__(self, a):
        super().__init__("vision")
        self.a = a
        self.patterns = load_patterns()
        colors = dict(self.patterns["colors"])
        colors["ball"] = self.patterns["ball"]
        self.cam = CameraModel.load(a.calib) if a.calib else None
        if a.calib:
            colors.update(yaml.safe_load(open(a.calib)).get("colors", {}))  # colours sampled on the real camera
        self.lut = ColorLUT(colors, max_de=a.max_de)
        self.decoder = None if self.cam is None else Decoder(self.patterns, self.lut, self.cam)
        self.tracker = Tracker(TrackParams())
        self.yolo = YoloThread(YoloRobots(a.weights, conf=a.conf), a.yolo_hz)
        self.pub = self.create_publisher(Field, "/field", 10)
        self.pub_img = self.create_publisher(Image, "/vision/image_annotated", 2)
        self.create_subscription(Image, a.image, self._on_image, qos_profile_sensor_data)
        if self.cam is None:
            self.tf = Buffer()
            self.tf_listener = TransformListener(self.tf, self)
            self.create_subscription(CameraInfo, a.camera_info, self._on_info, 1)
        self.frame, self.cv = None, threading.Condition()
        self.found = deque(maxlen=60)  # (frame stamp, pixels of the robots decoded in it): YOLO runs on these frames
        self.stats = {"frames": 0, "proc": 0.0, "age": 0.0, "robots": 0, "rois": 0, "t0": time.monotonic(), "yolo0": 0}

    def _on_info(self, info):
        if self.cam is not None:
            return
        try:
            tf = self.tf.lookup_transform("field", info.header.frame_id or self.a.camera_frame, Time())
        except Exception:
            return  # the static TF hasn't arrived yet
        tr, q = tf.transform.translation, tf.transform.rotation
        self.cam = CameraModel.from_ros(info, (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))
        self.decoder = Decoder(self.patterns, self.lut, self.cam)
        self.get_logger().info(f"camera model from camera_info + TF: {info.width}x{info.height}, "
                               f"camera at {np.round(self.cam.center, 3).tolist()} m")

    def _on_image(self, msg):  # rclpy thread: keep only the newest frame
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        rgb = img if msg.encoding == "rgb8" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.cv:
            self.frame = (stamp, msg.header.stamp, rgb)
            self.cv.notify()

    def run(self):
        while rclpy.ok():
            with self.cv:
                if self.frame is None:
                    self.cv.wait(0.5)
                if self.frame is None or self.decoder is None:
                    continue
                (t, stamp, rgb), self.frame = self.frame, None
            t0 = time.perf_counter()
            self.yolo.offer(t, rgb)
            robots, ball, rois = self.process(t, rgb)
            self.pub.publish(self.field_msg(t, stamp))
            s = self.stats
            s["frames"] += 1
            s["proc"] += time.perf_counter() - t0
            s["age"] += time.time() - t  # meaningful when stamps are wall time (the sim bridge, real cameras)
            s["robots"] += sum(o.id is not None for o in robots)
            s["rois"] += len(rois)
            if self.pub_img.get_subscription_count():
                self.publish_annotated(rgb, stamp, robots, ball, rois)
            self.log_stats()

    def process(self, t, rgb):
        cam, dec, top = self.cam, self.decoder, self.patterns["layout"]["top_height"]
        # windows: tracked robots predicted to this frame, plus YOLO boxes of robots nobody tracks yet
        size = self.patterns["layout"]["top_size"]
        rois = []
        for tr in self.tracker.fresh(t).values():
            dt = t - tr.t
            u, v = cam.to_pixel([tr.x.x[0] + tr.x.x[1] * dt, tr.y.x[0] + tr.y.x[1] * dt, top])
            # half-width = the top's width: a touching robot's ID squares then lie whole inside (cut by the
            # edge, their centroids move inwards until they can fit this robot's layout better than its own)
            r = size / cam.metres_per_pixel((u, v), top)
            rois.append((u - r, v - r, u + r, v + r))
        # YOLO's boxes are from an older frame; compare them with the robots decoded in that same frame
        # (extrapolating the tracks back misses robots that just started or stopped: a trailing box)
        t_yolo, boxes, _ = self.yolo.result
        known = next((uvs for ts, uvs in reversed(self.found) if ts == t_yolo), [])
        for x0, y0, x1, y1 in boxes:
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            r = size / cam.metres_per_pixel((u, v), top)  # the same window as a track's, on the box centre
            if all((u - ku) ** 2 + (v - kv) ** 2 > r * r / 4 for ku, kv in known):
                rois.append((u - r, v - r, u + r, v + r))
        robots = [o for o in (dec.robot(rgb, box) for box in rois) if o is not None]
        self.found.append((t, [o.uv for o in robots]))
        ball, b = None, self.tracker.ball
        if b is not None and t - b.seen < self.tracker.p.timeout:  # look near where it should be first
            dt = t - b.t
            u, v = cam.to_pixel([b.x.x[0] + b.x.x[1] * dt, b.y.x[0] + b.y.x[1] * dt, 0.02135])
            r = 0.15 / cam.metres_per_pixel((u, v))
            ball = dec.ball(rgb, near=self.tracker.ball_near(), window=(u - r, v - r, u + r, v + r))
        if ball is None:
            ball = dec.ball(rgb, near=self.tracker.ball_near())
        self.tracker.step(t, robots, None if ball is None else ball[:2])
        return robots, ball, rois

    def field_msg(self, t, stamp):
        f = Field()
        f.header.stamp, f.header.frame_id = stamp, "field"
        timeout = self.tracker.p.timeout
        b = self.tracker.ball
        if b is not None:
            f.ball = Object(id=0, detected=t - b.seen < timeout, x=b.x.x[0], y=b.y.x[0], vx=b.x.x[1], vy=b.y.x[1])
        for rid, tr in sorted(self.tracker.robots.items()):
            o = Object(id=rid, detected=t - tr.seen < timeout, x=tr.x.x[0], y=tr.y.x[0], theta=tr.theta.x[0],
                       vx=tr.x.x[1], vy=tr.y.x[1], w=tr.theta.x[1])
            (f.blue if rid <= 10 else f.yellow).append(o)
        return f

    def publish_annotated(self, rgb, stamp, robots, ball, rois):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for x0, y0, x1, y1 in rois:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (90, 90, 90), 1)
        top = self.patterns["layout"]["top_height"]
        for o in robots:
            u, v = (int(round(c)) for c in o.uv)
            col = (255, 200, 0) if o.team == "blue" else (0, 230, 255)
            cv2.circle(img, (u, v), 3, col, -1)
            if o.theta is not None:
                tip = self.cam.to_pixel([o.x + 0.05 * np.cos(o.theta), o.y + 0.05 * np.sin(o.theta), top])
                cv2.line(img, (u, v), (int(tip[0]), int(tip[1])), col, 2)
            cv2.putText(img, str(o.id) if o.id is not None else "?", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if ball is not None:
            cv2.circle(img, (int(ball[2][0]), int(ball[2][1])), 9, (0, 140, 255), 2)
        out = Image(height=img.shape[0], width=img.shape[1], encoding="bgr8", step=img.shape[1] * 3)
        out.header.stamp, out.header.frame_id = stamp, "overhead_camera_optical"
        out.data = img.tobytes()
        self.pub_img.publish(out)

    def log_stats(self):
        s = self.stats
        now = time.monotonic()
        if now - s["t0"] < 10 or not s["frames"]:
            return
        n, el = s["frames"], now - s["t0"]
        self.get_logger().info(f"{n / el:.1f} frames/s, YOLO {(self.yolo.runs - s['yolo0']) / el:.1f}/s, "
                               f"{1e3 * s['proc'] / n:.1f} ms per frame, {s['robots'] / n:.2f} robots decoded and {s['rois'] / n:.2f} windows per frame, "
                               f"frame age at publish {1e3 * s['age'] / n:.0f} ms")
        self.stats = {"frames": 0, "proc": 0.0, "age": 0.0, "robots": 0, "rois": 0, "t0": now, "yolo0": self.yolo.runs}


def _spin(node):
    try:
        rclpy.spin(node)
    except Exception:
        if rclpy.ok():
            raise  # a real error, not Ctrl+C shutting the context down


def main():
    ap = argparse.ArgumentParser(description="Overhead camera -> /field (see the module docstring)")
    ap.add_argument("--image", default="/overhead_camera/image_raw")
    ap.add_argument("--camera_info", default="/overhead_camera/camera_info")
    ap.add_argument("--camera_frame", default="overhead_camera_optical", help="TF frame if camera_info has none")
    ap.add_argument("--calib", help="camera calibration YAML (calibrate.py) instead of camera_info + TF")
    ap.add_argument("--weights", default=str(HERE / "models" / "robots_yolov8m.pt"))
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    ap.add_argument("--yolo_hz", type=float, default=10.0,
                    help="most YOLO runs per second: it finds new robots, tracks follow known ones every frame")
    ap.add_argument("--max_de", type=float, default=35.0, help="colour match tolerance (CIE Lab distance)")
    a = ap.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(args=sys.argv)
    node = Vision(a)
    threading.Thread(target=_spin, args=(node,), daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.yolo.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
