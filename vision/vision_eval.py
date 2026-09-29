#!/usr/bin/env python3
"""Score the vision (/field) against the sim's ground truth (/field_truth).

  python3 vision/vision_eval.py                 # prints a summary every 10 s, and at Ctrl+C
  python3 vision/vision_eval.py --duration 30   # one 30 s summary, then exit

Each /field message is paired with the /field_truth sample of the same stamp (both are stamped with
the sim time of the camera frame). Reported per window: how many robots were seen with the right id,
wrong ids (an id reported far from where that robot is, or one that isn't on the field), position /
heading / velocity errors of robots and ball, and the age of each frame when /field arrived
(meaningful while the sim runs in real time).
"""
import argparse
import math
import sys
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from vsss_msgs.msg import Field

WRONG = 0.05  # [m] an id reported this far from its robot counts as wrong


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _objects(f):
    return {o.id: o for o in list(f.blue) + list(f.yellow)}


class Eval(Node):
    def __init__(self, a):
        super().__init__("vision_eval")
        self.a = a
        self.truth = deque(maxlen=500)
        self.create_subscription(Field, "/field_truth", lambda m: self.truth.append((_stamp(m), m)), 50)
        self.create_subscription(Field, "/field", self._on_field, 50)
        self.reset()
        self.t_start = self.t_window = time.monotonic()

    def reset(self):
        self.n = dict(msgs=0, unmatched=0, robots=0, seen=0, wrong=0, balls=0, ball_seen=0)
        self.err = {k: [] for k in ("pos", "theta", "vel", "w", "ball", "ball_vel", "age", "speed")}

    def _on_field(self, f):
        t = _stamp(f)
        self.err["age"].append(time.time() - t)
        match = min(self.truth, key=lambda s: abs(s[0] - t), default=None)
        if match is None or abs(match[0] - t) > 0.002:
            self.n["unmatched"] += 1
            return
        truth, seen = _objects(match[1]), _objects(f)
        self.n["msgs"] += 1
        for rid, tr in truth.items():
            self.n["robots"] += 1
            o = seen.get(rid)
            if o is None or not o.detected:
                continue
            d = math.hypot(o.x - tr.x, o.y - tr.y)
            if d > WRONG:
                self.n["wrong"] += 1
                continue
            self.n["seen"] += 1
            self.err["speed"].append(math.hypot(tr.vx, tr.vy))
            self.err["pos"].append(d)
            self.err["theta"].append(abs((o.theta - tr.theta + math.pi) % (2 * math.pi) - math.pi))
            self.err["vel"].append(math.hypot(o.vx - tr.vx, o.vy - tr.vy))
            self.err["w"].append(abs(o.w - tr.w))
        self.n["wrong"] += sum(1 for rid, o in seen.items() if o.detected and rid not in truth)
        self.n["balls"] += 1
        b, tb = f.ball, match[1].ball
        if b.detected:
            self.n["ball_seen"] += 1
            self.err["ball"].append(math.hypot(b.x - tb.x, b.y - tb.y))
            self.err["ball_vel"].append(math.hypot(b.vx - tb.vx, b.vy - tb.vy))
        now = time.monotonic()
        if self.a.duration is None and now - self.t_window > 10:
            self.report()
            self.reset()
            self.t_window = now

    def report(self):
        n, e = self.n, {k: np.array(v) for k, v in self.err.items()}
        if not n["msgs"]:
            print("no /field messages matched to /field_truth yet" + (f" ({n['unmatched']} unmatched)" if n["unmatched"] else ""))
            return
        q = lambda a, s=1.0: f"{np.mean(a) * s:.1f} / {np.percentile(a, 95) * s:.1f}" if len(a) else "-"
        print(f"{n['msgs']} frames ({n['unmatched']} without truth); robots seen with the right id "
              f"{100 * n['seen'] / max(1, n['robots']):.1f} %, wrong ids {n['wrong']}; ball seen "
              f"{100 * n['ball_seen'] / max(1, n['balls']):.1f} %")
        print(f"  mean / 95 %: robot position {q(e['pos'], 1e3)} mm, heading {q(e['theta'], 180 / math.pi)} deg, "
              f"velocity {q(e['vel'], 1e3)} mm/s, turn rate {q(e['w'])} rad/s (robots moved at {q(e['speed'])} m/s)")
        print(f"               ball position {q(e['ball'], 1e3)} mm, velocity {q(e['ball_vel'], 1e3)} mm/s; "
              f"frame age at /field {q(e['age'], 1e3)} ms")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Score /field against /field_truth")
    ap.add_argument("--duration", type=float, help="collect this many seconds, print one summary, exit")
    a = ap.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(args=sys.argv)
    node = Eval(a)
    try:
        while rclpy.ok() and (a.duration is None or time.monotonic() - node.t_start < a.duration):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
