"""Camera model: pixels <-> field coordinates on a horizontal plane at a chosen height.

The field frame is the sim's: origin at the field centre, x towards the yellow goal, y left, z up
[m]. A point on a robot top is found by intersecting its pixel's ray with the plane z = top height,
which removes the parallax of markers 7 cm above the floor (up to ~3 cm near the goals at 2 m).

Sources:
  * from_ros(camera_info, pose): intrinsics from sensor_msgs/CameraInfo, pose from the TF
    field -> camera optical frame (what the sim publishes);
  * load(yaml): a calibration file written by calibrate.py for a real camera.
"""
import cv2
import numpy as np
import yaml


class CameraModel:
    def __init__(self, K, dist, R, t, size):
        """K 3x3, dist (k1 k2 p1 p2 k3), R and t: field -> camera (x_cam = R x_field + t), size (w, h)."""
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=float).ravel()
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.t = np.asarray(t, dtype=float).ravel()
        self.size = tuple(int(v) for v in size)
        self.center = -self.R.T @ self.t  # camera position in the field frame

    @classmethod
    def from_ros(cls, info, translation, quat_xyzw):
        """camera_info + the pose of the optical frame in the field frame (TF field -> optical)."""
        x, y, z, w = quat_xyzw
        R_fo = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # optical -> field
        p = np.asarray(translation, dtype=float)
        dist = list(info.d) if len(info.d) else [0.0] * 5
        return cls(np.array(info.k).reshape(3, 3), dist, R_fo.T, -R_fo.T @ p, (info.width, info.height))

    @classmethod
    def load(cls, path):
        c = yaml.safe_load(open(path))
        R, _ = cv2.Rodrigues(np.array(c["rvec"], dtype=float))
        return cls(c["K"], c["dist"], R, c["tvec"], c["size"])

    def save(self, path, extra=None):
        rvec, _ = cv2.Rodrigues(self.R)
        data = {"size": list(self.size), "K": self.K.tolist(), "dist": self.dist.tolist(),
                "rvec": rvec.ravel().tolist(), "tvec": self.t.tolist()}
        data.update(extra or {})
        with open(path, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=None, sort_keys=False)

    def to_field(self, uv, z=0.0):
        """Pixels (N x 2, or one (u, v)) -> field (x, y) on the plane at height z."""
        uv = np.asarray(uv, dtype=float)
        single = uv.ndim == 1
        n = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([n, np.ones(len(n))]) @ self.R  # R.T @ [x, y, 1] for each row
        s = (z - self.center[2]) / rays[:, 2]
        xy = self.center[:2] + s[:, None] * rays[:, :2]
        return xy[0] if single else xy

    def to_pixel(self, xyz):
        """Field points (N x 3, or one) -> pixels."""
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        rvec, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, self.t, self.K, self.dist)
        uv = uv.reshape(-1, 2)
        return uv[0] if single else uv

    def metres_per_pixel(self, uv, z=0.0):
        """Size of one pixel on the plane at height z, around pixel uv."""
        u, v = uv
        a, b = self.to_field([[u - 0.5, v], [u + 0.5, v]], z)
        return float(np.linalg.norm(b - a))
