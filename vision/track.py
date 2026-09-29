"""Tracking: a constant-velocity Kalman filter per robot id and one for the ball.

Like the old vision's Kalman class (state x, y, theta, vx, vy, w), but in SI units, timed by the
camera frames' stamps rather than wall time, with the heading's innovation wrapped, and split into
independent (value, rate) filters, which is the same model with a block-diagonal covariance.
"""
import math
from dataclasses import dataclass, field

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class KF2:
    """(value, rate) with white-noise acceleration of std sigma_a."""

    def __init__(self, z, r, sigma_a, angle=False):
        self.x = np.array([z, 0.0])
        self.P = np.diag([r * r, 1.0])
        self.q, self.angle = sigma_a ** 2, angle

    def predict(self, dt):
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([dt * dt / 2, dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * np.outer(G, G)
        if self.angle:
            self.x[0] = _wrap(self.x[0])

    def update(self, z, r):
        y = z - self.x[0]
        if self.angle:
            y = _wrap(y)
        s = self.P[0, 0] + r * r
        k = self.P[:, 0] / s
        self.x = self.x + k * y
        self.P = self.P - np.outer(k, self.P[0, :])
        if self.angle:
            self.x[0] = _wrap(self.x[0])
        return abs(y), s


@dataclass
class TrackParams:
    sigma_xy: float = 0.003  # [m] robot position measurement noise (full decode)
    sigma_xy_partial: float = 0.02  # [m] team patch only: the centre is ~2 cm off in an unknown direction
    sigma_theta: float = 0.03  # [rad]
    sigma_ball: float = 0.004  # [m]
    accel: float = 6.0  # [m/s^2] robots (traction limit ~7)
    alpha: float = 60.0  # [rad/s^2]
    ball_accel: float = 15.0  # [m/s^2] the ball changes speed abruptly on hits
    gate: float = 0.25  # [m] a decoded id this far from its track is a misread (unless it keeps happening)
    min_separation: float = 0.05  # [m] robots are 7.5 cm wide: a new track this close to another is that robot misread
    timeout: float = 0.25  # [s] not seen for this long: detected = false


@dataclass
class Track:
    x: KF2
    y: KF2
    theta: KF2
    t: float  # time of the filter state
    seen: float  # last update
    misses: int = field(default=0)  # consecutive gated observations

    def predict(self, t):
        for f in (self.x, self.y, self.theta):
            f.predict(t - self.t)
        self.t = max(self.t, t)


class Tracker:
    def __init__(self, params: TrackParams = None):
        self.p = params or TrackParams()
        self.robots = {}  # id -> Track (team from the id: 1-10 blue, 11-20 yellow)
        self.ball = None

    def fresh(self, t):
        return {i: tr for i, tr in self.robots.items() if t - tr.seen < self.p.timeout}

    def step(self, t, robots, ball):
        """Advance to frame time t and apply its observations (vision/detect.py RobotObs, ball (x, y))."""
        p = self.p
        for tr in self.robots.values():
            tr.predict(t)
        if self.ball is not None:
            self.ball.predict(t)
        done = set()
        for o in sorted((o for o in robots if o.id is not None), key=lambda o: o.id):
            tr = self.robots.get(o.id)
            if o.id in done:
                continue
            if (tr is None or t - tr.seen >= p.timeout or tr.misses >= 3) and self._taken(t, o.id, o.x, o.y):
                continue  # a new track would sit on another robot's: it's that robot, misread
            if tr is None:
                self.robots[o.id] = Track(KF2(o.x, p.sigma_xy, p.accel), KF2(o.y, p.sigma_xy, p.accel),
                                          KF2(o.theta, p.sigma_theta, p.alpha, angle=True), t, t)
            else:
                far = math.hypot(o.x - tr.x.x[0], o.y - tr.y.x[0]) > p.gate
                if far and t - tr.seen < p.timeout and tr.misses < 3:
                    tr.misses += 1  # probably another robot read as this id
                    continue
                if far:  # it really moved (or was lost): start over there
                    self.robots[o.id] = Track(KF2(o.x, p.sigma_xy, p.accel), KF2(o.y, p.sigma_xy, p.accel),
                                              KF2(o.theta, p.sigma_theta, p.alpha, angle=True), t, t)
                else:
                    tr.x.update(o.x, p.sigma_xy)
                    tr.y.update(o.y, p.sigma_xy)
                    tr.theta.update(o.theta, p.sigma_theta)
                    tr.seen, tr.misses = t, 0
            done.add(o.id)
        # team patch only: a position update for the nearest fresh, not yet updated track of that team
        for o in (o for o in robots if o.id is None):
            first = 1 if o.team == "blue" else 11
            cands = [(math.hypot(o.x - tr.x.x[0], o.y - tr.y.x[0]), i) for i, tr in self.fresh(t).items()
                     if first <= i < first + 10 and i not in done]
            if cands and min(cands)[0] < 0.05:
                i = min(cands)[1]
                self.robots[i].x.update(o.x, p.sigma_xy_partial)
                self.robots[i].y.update(o.y, p.sigma_xy_partial)
                self.robots[i].seen = t
                done.add(i)
        if ball is not None:
            if self.ball is None or t - self.ball.seen > p.timeout:
                self.ball = Track(KF2(ball[0], p.sigma_ball, p.ball_accel), KF2(ball[1], p.sigma_ball, p.ball_accel),
                                  KF2(0.0, 1.0, 0.0), t, t)
            else:
                self.ball.x.update(ball[0], p.sigma_ball)
                self.ball.y.update(ball[1], p.sigma_ball)
                self.ball.seen = t

    def _taken(self, t, rid, x, y):
        return any(i != rid and math.hypot(x - tr.x.x[0], y - tr.y.x[0]) < self.p.min_separation
                   for i, tr in self.fresh(t).items())

    def ball_near(self):
        return None if self.ball is None else (self.ball.x.x[0], self.ball.y.x[0])
