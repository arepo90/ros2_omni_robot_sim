#!/usr/bin/env python3
"""Keyboard teleop for the VSSS sim, in the terminal: the robot moves only while keys are held.

  python3 teleop.py                          # drives blue_0
  python3 teleop.py --robot yellow_1 --speed 0.6 --turn 5

  W / S         forward / backward
  A / D         left / right (strafe)
  Q / E         turn counter-clockwise / clockwise
  Up / Down     speed +/- 10 % of --speed and --turn (starts at --percent, 50 %)
  Esc, Ctrl+C   quit

Keys combine: W+A is a diagonal at the set speed, W+Q an arc; opposite keys cancel.

How it knows a key is held: a terminal only receives characters, and auto-repeat starts 0.5 s
after the first one, so from characters a tap looks like the start of a hold, and only the last
key repeats. Instead this polls the X server for the keys physically down (XQueryKeymap) at --rate,
which is exact for any combination of keys. Consequences:
  * X11 only (not Wayland), which is what this laptop runs;
  * the X keymap is global, so keys only count while the window teleop was started from (its
    terminal) has focus; switching to another window stops the robot. Other tabs of the same
    terminal window still count;
  * the terminal's echo is off while it runs, and typed characters are discarded, so the keys
    don't end up in your shell afterwards.

Releasing the last key sends a zero command on the next poll (plus a few repeats). The bridge's
duty ramp (ros_bridge.py --ramp, 0.2 s by default) smooths starts and stops like the firmware, so a
short tap moves the robot a small step.
"""
import argparse
import math
import os
import select
import signal
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from Xlib import XK, display as xdisplay
from Xlib.error import DisplayError

# key -> (x, y, yaw) direction in the robot frame: x forward, y left, yaw counter-clockwise
MOVE = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "q": (0, 0, 1), "e": (0, 0, -1)}
ZERO_REPEATS = 3  # zero commands sent after the last key is released, in case one is lost


class Keyboard:
    """The keys physically down, read from the X server; none while another window has focus."""

    def __init__(self, names):
        self.d = xdisplay.Display()
        self.code = {n: self.d.keysym_to_keycode(XK.string_to_keysym(n)) for n in names}
        self.root = self.d.screen().root
        self.active_atom = self.d.intern_atom("_NET_ACTIVE_WINDOW")
        self.home = self._active()  # the terminal teleop was started from has focus right now

    def _active(self):
        prop = self.root.get_full_property(self.active_atom, 0)
        return prop.value[0] if prop is not None and len(prop.value) else None

    def focused(self):
        return self._active() == self.home

    def down(self):
        keymap = self.d.query_keymap()  # 256-bit map, one bit per keycode
        return {n for n, c in self.code.items() if c and keymap[c // 8] & (1 << (c % 8))}


def command(held, percent, a):
    """(vx, vy, wz) for the held keys. Diagonals keep the set speed; opposite keys cancel."""
    x, y, z = (sum(MOVE[k][i] for k in held) for i in range(3))
    scale = percent / 100
    n = math.hypot(x, y)
    vx, vy = (x / n * a.speed * scale, y / n * a.speed * scale) if n else (0.0, 0.0)
    return vx, vy, z * a.turn * scale


def run(a, kb, pub, out=sys.stdout):
    """Poll keys and publish until Esc. Returns on Esc; Ctrl+C raises KeyboardInterrupt."""
    def publish(vx, vy, wz):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        pub.publish(t)

    percent, prev, zeros, shown = a.percent, set(), 0, None
    period = 1.0 / a.rate
    tty_in = sys.stdin.isatty()
    while True:
        t0 = time.monotonic()
        focused = kb.focused()
        keys = kb.down() if focused else set()
        if "Escape" in keys:
            return
        for key, sign in (("Up", 1), ("Down", -1)):
            if key in keys and key not in prev:  # one step per press
                percent = min(100, max(a.step, percent + sign * a.step))
        held = keys & MOVE.keys()
        vx, vy, wz = command(held, percent, a)
        if held:
            publish(vx, vy, wz)
            zeros = ZERO_REPEATS
        elif zeros > 0:  # just released, or focus moved away: stop now
            publish(0.0, 0.0, 0.0)
            zeros -= 1
        prev = keys
        s = percent / 100
        state = " ".join(k.upper() for k in "wasdqe" if k in held) or "-"
        line = (f"{a.robot}  speed {percent:3d} % ({a.speed * s:.2f} m/s, {a.turn * s:.1f} rad/s)  "
                f"cmd vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}  keys {state:11s}"
                f"{'' if focused else '  (not focused: click this terminal)'}")
        if line != shown:  # redraw the status line only when it changes
            out.write(f"\r{line}\x1b[K")
            out.flush()
            shown = line
        while tty_in and select.select([sys.stdin], [], [], 0)[0]:  # discard what the terminal typed
            if not os.read(sys.stdin.fileno(), 1024):
                tty_in = False  # end of input (terminal gone): stop reading
        time.sleep(max(0.0, period - (time.monotonic() - t0)))


def _stop(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description="Hold-to-move keyboard teleop (see the module docstring)")
    ap.add_argument("--robot", default="blue_0", help="robot namespace: publishes /<robot>/cmd_vel")
    ap.add_argument("--speed", type=float, default=0.8, help="linear speed at 100 %% [m/s]")
    ap.add_argument("--turn", type=float, default=6.0, help="turn rate at 100 %% [rad/s]")
    ap.add_argument("--percent", type=int, default=50, help="starting speed percentage")
    ap.add_argument("--step", type=int, default=10, help="percentage change per Up/Down press")
    ap.add_argument("--rate", type=float, default=50.0, help="key polling and publish rate [Hz]")
    a = ap.parse_args(remove_ros_args(sys.argv)[1:])
    try:
        kb = Keyboard(list(MOVE) + ["Up", "Down", "Escape"])
    except DisplayError as e:
        sys.exit(f"teleop needs an X11 display (DISPLAY={os.environ.get('DISPLAY')}): {e}")

    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    # Ctrl+C, a closed terminal or kill all end in the finally below, which sends the robot a stop.
    # Set explicitly: a process started in the background or from a script may inherit SIGINT ignored.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)
    node = rclpy.create_node(f"teleop_{a.robot}")
    pub = node.create_publisher(Twist, f"/{a.robot}/cmd_vel", 10)
    print(f"teleop -> /{a.robot}/cmd_vel   W/S forward/back  A/D left/right  Q/E turn  "
          f"Up/Down speed  Esc quit\nKeys act only while held and this terminal has focus.")
    saved = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if saved:
        tty.setcbreak(sys.stdin)  # no echo, no line buffering
    try:
        run(a, kb, pub)
    except KeyboardInterrupt:
        pass
    finally:
        pub.publish(Twist())  # never leave the robot with a command
        if saved:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
