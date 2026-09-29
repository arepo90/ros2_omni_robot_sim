"""Detection: YOLO finds robots; their colour patches give team, id, position and heading; plus the ball.

Follows the old vision (vsss repo, commit 7490988, vision_general.py): a YOLO box per robot, then in
it the team patch and the two ID squares; heading from the team patch towards the ID squares,
left / right by a cross product, id from the pattern table. Changes:
  * colours come from a lookup table built from reference colours (like the old lut_*.npy, but
    generated from vision/patterns.yaml, or from calibrated colours for a real camera);
  * the geometry is done in field coordinates on the plane of the robot tops, so heading and left /
    right don't depend on image axes, and the camera's parallax is removed;
  * the two ID squares are the pair of ID-colour blobs that best fits the known top layout, so a
    neighbouring robot's patches in the same box aren't taken for them.
"""
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PATTERNS_PATH = Path(__file__).with_name("patterns.yaml")
# colours that are not patches: nearest to one of these means "no patch" (the sim's floor, walls,
# lines and surroundings; a real black field is similar)
BACKGROUND = {"field": (10, 10, 10), "wall": (20, 20, 20), "line": (255, 255, 255), "surround": (90, 90, 92)}


def load_patterns(path=PATTERNS_PATH):
    p = yaml.safe_load(open(path))
    lookup = {}  # (team, left, right) -> id
    for team, first in (("blue", 1), ("yellow", 11)):
        for k, (left, right) in enumerate(p["ids"]):
            lookup[(team, left, right)] = first + k
    p["lookup"] = lookup
    p["team_of_color"] = {c: t for t, c in p["teams"].items()}
    return p


class ColorLUT:
    """RGB -> label: the nearest reference colour in CIE Lab, or 0 ("none") if the nearest is a
    background colour or farther than max_de. 64 levels per channel, so labelling is one lookup."""

    def __init__(self, colors, background=BACKGROUND, max_de=35.0):
        self.names = ["none"] + list(colors)
        levels = (np.arange(64) << 2) + 2
        r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
        grid = np.stack([b, g, r], -1).reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(grid, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        refs = [c for c in colors.values()] + [c for c in background.values()]
        ref_lab = cv2.cvtColor(np.uint8([[c[::-1] for c in refs]]), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
        dist = np.linalg.norm(lab[:, None, :] - ref_lab[None], axis=2)
        nearest = dist.argmin(1)
        ok = (nearest < len(colors)) & (dist.min(1) < max_de)
        self.lut = np.where(ok, nearest + 1, 0).astype(np.uint8).reshape(64, 64, 64)

    def label(self, rgb):
        return self.lut[rgb[..., 0] >> 2, rgb[..., 1] >> 2, rgb[..., 2] >> 2]

    def index(self, name):
        return self.names.index(name)


@dataclass
class RobotObs:
    id: object  # int, or None if only the team patch was found
    team: str
    x: float
    y: float
    theta: object  # float [rad], or None
    uv: tuple  # pixel of the centre, for drawing


class Decoder:
    """Reads robots (in image regions) and the ball (anywhere) from a labelled image."""

    def __init__(self, patterns, lut: ColorLUT, cam):
        self.p, self.lut, self.cam = patterns, lut, cam
        lay = patterns["layout"]
        self.z_top = lay["top_height"]
        inner = lay["top_size"] - 2 * lay["border"]
        self.square = (inner - lay["gap"]) / 2  # side of an ID square [m]
        self.spacing = self.square + lay["gap"]  # ID square centres apart, and team patch -> their midpoint
        self.team_labels = {lut.index(c): t for c, t in patterns["team_of_color"].items()}
        self.id_labels = {lut.index(c): c for c in set(c for pair in patterns["ids"] for c in pair)}
        self.ball_label = lut.index("ball")

    def _blobs(self, labels, x0, y0, wanted, min_area):
        """Blobs of the wanted labels: (label, area, u, v) in full-image pixels."""
        out = []
        for lab in wanted:
            mask = (labels == lab).astype(np.uint8)
            if not mask.any():
                continue
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] >= min_area:
                    out.append((lab, stats[k, cv2.CC_STAT_AREA], cents[k][0] + x0, cents[k][1] + y0))
        return out

    def robot(self, rgb, box):
        """Decode one robot in box (x0, y0, x1, y1) of the image; None if no team patch."""
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        roi = self.lut.label(rgb[y0:y1, x0:x1])
        centre_uv = ((x0 + x1) / 2, (y0 + y1) / 2)
        mpp = self.cam.metres_per_pixel(centre_uv, self.z_top)
        min_area = max(4, 0.2 * (self.square / mpp) ** 2)
        teams = self._blobs(roi, x0, y0, self.team_labels, min_area)
        if not teams:
            return None
        ids = self._blobs(roi, x0, y0, self.id_labels, min_area)
        pts = self.cam.to_field([(u, v) for _, _, u, v in teams + ids], self.z_top)
        tpts, ipts = pts[:len(teams)], pts[len(teams):]
        c = self.cam.to_field(centre_uv, self.z_top)
        k = int(np.argmin(np.linalg.norm(tpts - c, axis=1)))  # the team patch nearest the box centre
        team, tp = self.team_labels[teams[k][0]], tpts[k]
        best, best_err = None, 0.4 * self.spacing  # the ID pair that fits the layout best
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                mid = (ipts[i] + ipts[j]) / 2
                err = abs(np.linalg.norm(ipts[i] - ipts[j]) - self.spacing) + abs(np.linalg.norm(mid - tp) - self.spacing)
                if err < best_err:
                    best, best_err = (i, j), err
        if best is None:  # team patch only: position, no heading or id
            return RobotObs(None, team, float(tp[0]), float(tp[1]), None, tuple(self.cam.to_pixel([*tp, self.z_top])))
        i, j = best
        mid = (ipts[i] + ipts[j]) / 2
        heading = mid - tp
        theta = math.atan2(heading[1], heading[0])
        side = lambda q: heading[0] * (q[1] - mid[1]) - heading[1] * (q[0] - mid[0])  # > 0: on the robot's left
        (left, right) = (i, j) if side(ipts[i]) > 0 else (j, i)
        rid = self.p["lookup"].get((team, self.id_labels[ids[left][0]], self.id_labels[ids[right][0]]))
        centre = (tp + mid) / 2
        return RobotObs(rid, team, float(centre[0]), float(centre[1]), theta,
                        tuple(self.cam.to_pixel([*centre, self.z_top])))

    def ball(self, rgb, near=None, radius=0.02135, window=None):
        """The ball's field (x, y) and pixel, or None: the orange blob of about the right size,
        nearest `near` (the last estimate) if given, else the largest. window (x0, y0, x1, y1)
        limits the search (around the predicted ball) and saves most of the time."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (0, 0, w, h) if window is None else (
            max(0, int(window[0])), max(0, int(window[1])), min(w, int(window[2])), min(h, int(window[3])))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        labels = self.lut.label(rgb[y0:y1, x0:x1])
        mask = cv2.morphologyEx((labels == self.ball_label).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        cents = cents + (x0, y0)
        mpp = self.cam.metres_per_pixel(((x0 + x1) / 2, (y0 + y1) / 2), radius)
        expected = math.pi * (radius / mpp) ** 2
        cand = [k for k in range(1, n) if 0.25 * expected <= stats[k, cv2.CC_STAT_AREA] <= 4 * expected]
        if not cand:
            return None
        xy = self.cam.to_field([cents[k] for k in cand], radius)  # ball centre height
        if near is not None:
            d = np.linalg.norm(xy - np.asarray(near), axis=1)
            k = int(np.argmin(d))
        else:
            k = int(np.argmax([stats[c, cv2.CC_STAT_AREA] for c in cand]))
        return float(xy[k][0]), float(xy[k][1]), tuple(cents[cand[k]])


class YoloRobots:
    """The robot detector: YOLOv8 with one class, "robot" (the old vision's model).

    Same letterbox, network, NMS and box scaling as ultralytics' predictor (`YOLO(weights)(bgr)`),
    without its per-call overhead: on a GPU the fused network runs in fp16, recorded once per image
    size as a CUDA graph, and a call waits for the GPU only once, asleep. ~13 ms on the RTX 4060
    (the predictor: 23 ms; with torch 2.0.1, which has no sm_89 kernels: 44 ms), and the GIL is
    free meanwhile (every wait for the GPU releases it, and getting it back can take the 5 ms
    switch interval while the frame loop runs)."""

    def __init__(self, weights, conf=0.3, iou=0.7, imgsz=640, device=None):
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never pip-install anything at run time
        os.environ.setdefault("YOLO_OFFLINE", "True")
        import torch
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        self.torch = torch
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        net = YOLO(str(weights)).model.fuse(verbose=False).to(self.device).eval()
        self.net = net.half() if self.cuda else net.float()
        self.letterbox = LetterBox((imgsz, imgsz), auto=True, stride=int(self.net.stride.max()))  # as the predictor
        self.conf, self.iou = conf, iou
        self.shape, self.graph = None, None
        if self.cuda:
            self.done = torch.cuda.Event(blocking=True)  # synchronize() sleeps instead of spinning

    def _forward(self):
        x = self.inp.flip(-1).permute(2, 0, 1)[None]  # BGR HWC uint8 -> RGB NCHW (no index tensor: graph-safe)
        x = x.half() if self.cuda else x.float()
        return self.net(x / 255)[0]  # (1, 4 + classes, anchors): box centre, size, class scores

    def _prepare(self, shape):
        """Buffers for this letterboxed size and, on a GPU, the CUDA graph that reads them."""
        torch = self.torch
        self.shape, self.graph = shape, None
        self.inp = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        self.inp_host = torch.zeros(shape, dtype=torch.uint8, pin_memory=self.cuda)
        with torch.inference_mode():
            if self.cuda:
                try:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(3):  # warm-up (cuDNN picks its kernels) before recording
                            self._forward()
                    torch.cuda.current_stream().wait_stream(s)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self.out = self._forward()
                    self.graph = graph
                except Exception as e:  # run it eagerly then: same results, more CPU time per call
                    print(f"YoloRobots: no CUDA graph ({e}); running the network eagerly", file=sys.stderr)
            out = self._forward()
        self.out_host = torch.zeros(out.shape, dtype=out.dtype, pin_memory=self.cuda)
        self.inp_np, self.out_np = self.inp_host.numpy(), self.out_host[0].numpy()  # views of the same memory

    def __call__(self, bgr):
        """Boxes (N x 4, x0 y0 x1 y1, pixels of `bgr`) and confidences, best first."""
        from ultralytics.utils.ops import scale_boxes
        img = self.letterbox(image=bgr)
        if img.shape != self.shape:
            self._prepare(img.shape)
        # few torch calls: each one releases the GIL, and under a busy frame loop getting it back costs
        # up to the 5 ms switch interval, so the rest is numpy
        with self.torch.inference_mode():
            self.inp_np[:] = img
            self.inp.copy_(self.inp_host, non_blocking=True)
            if self.graph is not None:
                self.graph.replay()
                out = self.out
            else:
                out = self._forward()
            self.out_host.copy_(out, non_blocking=True)
            if self.cuda:
                self.done.record()
                self.done.synchronize()  # the only wait for the GPU
        # NMS as ultralytics' (classes=[0]): the anchor's best class must be 0, with score > conf
        p = self.out_np.astype(np.float32)
        scores = p[4:]
        keep = (scores.argmax(0) == 0) & (scores[0] > self.conf)
        xywh, conf = p[:4, keep].T, scores[0, keep]
        xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        k = _nms(xyxy, conf, self.iou)[:300]
        return scale_boxes(img.shape[:2], xyxy[k], bgr.shape[:2]), conf[k]


def _nms(xyxy, scores, iou):
    """Greedy non-maximum suppression, as torchvision.ops.nms: indices kept, best first."""
    order = np.argsort(-scores, kind="stable")
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    kept = []
    while order.size:
        i, rest = order[0], order[1:]
        kept.append(i)
        w = np.clip(np.minimum(xyxy[i, 2], xyxy[rest, 2]) - np.maximum(xyxy[i, 0], xyxy[rest, 0]), 0, None)
        h = np.clip(np.minimum(xyxy[i, 3], xyxy[rest, 3]) - np.maximum(xyxy[i, 1], xyxy[rest, 1]), 0, None)
        inter = w * h
        order = rest[inter / (area[i] + area[rest] - inter) <= iou]
    return np.array(kept, dtype=int)
