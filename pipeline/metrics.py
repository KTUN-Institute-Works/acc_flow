"""
DUTCode / NNDVS ile aynı stabilizasyon metrikleri (evaluation/analyzer.py -> MetricAnalyzer):

  Crop Ratio       (↑)  : stabilize karede kalan geçerli alan oranı (ortalama ve minimum)
  Distortion       (↑)  : homografinin anizotropisi s_min/s_max (1.0 = bozulma yok), en kötü kare
  Stability Score  (↑)  : stabilize videodaki kamera yolunun düşük frekans enerji oranı (KLT + homografi)

Eski tek-video evaluation/metrics.py betiğindeki yöntemle aynı: orijinal kare ile stabilize kare arasında Farneback optik akışı
hesaplanır, akış ızgarası üzerinden RANSAC homografisi bulunur. Bellek için akış haritası
hesaplanırken örneklenir (tam çözünürlüklü tüm akışı saklamak klip başına GB'larca RAM ister).
"""
import os
import sys

import cv2
import numpy as np

import config

_EVAL_DIR = os.path.join(config.PROJECT_ROOT, "evaluation")
for p in (config.PROJECT_ROOT, _EVAL_DIR):
    if p not in sys.path:
        sys.path.append(p)

from analyzer import MetricAnalyzer  # noqa: E402  (evaluation/analyzer.py)
from utils import getBaseGrid, RunningAverage  # noqa: E402,F401


class SampledMetricAnalyzer(MetricAnalyzer):
    """MetricAnalyzer ile birebir aynı hesap; akış haritası önceden `scale_factor` adımıyla örneklenmiş gelir.
    Ek olarak minimum crop ratio'yu da döndürür."""

    def preprocess(self, flowmap_array, video_path):
        s = self.scale_factor
        # base_grid zaten [::s, ::s] örneklenmiş; akış da öyle geldiği için tekrar örneklemeyi atla
        h, w = self.base_grid.shape[:2]
        homo_array = []
        for i in range(flowmap_array.shape[0]):
            fm = flowmap_array[i][:h, :w]
            homo, _ = cv2.findHomography(self.base_grid.reshape((h * w, 1, 2)),
                                         (self.base_grid + fm).reshape(h * w, 1, 2), cv2.RANSAC)
            if homo is None:
                homo = np.eye(3)
            homo_array.append(homo)

        capture = cv2.VideoCapture(video_path)
        self.tracker.initialize()
        self.estimater.initialize()
        H = np.eye(3, dtype=np.float32)
        path_array = [H.copy()]
        ok, src_frame = capture.read()
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(n):
            ok, dst_frame = capture.read()
            if not ok:
                break
            pts_src, pts_dst = self.tracker.track_features(src_frame, dst_frame)
            if pts_src.shape[0] < 4:
                continue
            motion = self.estimater.estimate_motion(pts_src, pts_dst)
            if motion is None:
                continue
            H = np.matmul(motion, H)
            path_array.append(H.copy())
            src_frame = dst_frame
        capture.release()
        return np.float32(homo_array), np.float32(path_array)

    def run(self, flowmap_array, video_path):
        homo_array, path_array = self.preprocess(flowmap_array, video_path)
        crop, min_cr, distortion = [], 1e6, 1e6
        for i in range(self.start, homo_array.shape[0]):
            cr = self.findAreaPersentAfterWarp(homo_array[i], self.frame_width, self.frame_height)
            crop.append(cr)
            min_cr = min(min_cr, cr)
            distortion = min(distortion, self.findAnisotropicAfterWarp(homo_array[i]))
        stab = self.findLowFrequencyPersentAfterWarp(path_array)
        return {
            "CropRatio": float(np.mean(crop)),
            "MinCropRatio": float(min_cr),
            "Distortion": float(distortion),
            "Stability": float(stab),
        }


def sampled_warp_flow(original_video, stabilized_video, step):
    """Eski evaluation/metrics.py::compute_warp_flowmap ile aynı (Farneback orijinal -> stabilize),
    ancak her akış haritası [::step, ::step] örneklenerek saklanır."""
    cap_o = cv2.VideoCapture(original_video)
    cap_s = cv2.VideoCapture(stabilized_video)
    ok1, fo = cap_o.read()
    ok2, fs = cap_s.read()
    if not (ok1 and ok2):
        raise RuntimeError("Videolar okunamadı: {} / {}".format(original_video, stabilized_video))
    flows = []
    while True:
        ok1, fo = cap_o.read()
        ok2, fs = cap_s.read()
        if not (ok1 and ok2):
            break
        go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
        if go.shape != gs.shape:
            gs = cv2.resize(gs, (go.shape[1], go.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(go, gs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow[::step, ::step].copy())
    cap_o.release()
    cap_s.release()
    return np.asarray(flows, np.float32)


def dut_metrics(original_video, stabilized_video, step=config.METRIC_GRID_STEP):
    cap = cv2.VideoCapture(original_video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    flows = sampled_warp_flow(original_video, stabilized_video, step)
    analyzer = SampledMetricAnalyzer(frame_width=W, frame_height=H, scale_factor=step)
    return analyzer.run(flows, stabilized_video)
