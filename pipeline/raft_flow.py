"""
RAFT ile kareler arası global kamera kayması (stage3 + stage4'ün ilk yarısı).

Tam optik akış haritasını diske yazmak yerine (klip başına ~1 GB) her kare çifti için
yalnızca medyan (dx, dy) değerini tutar; stage4 zaten sadece bunu kullanıyordu.
"""
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np

from pipeline.device import get_device
import torch

import config

if config.RAFT_DIR not in sys.path:
    sys.path.append(config.RAFT_DIR)

from core.raft import RAFT  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


def _sample_points(gray, step, grad_quantile):
    """Akışın örnekleneceği noktalar.
    "corners" : Shi-Tomasi köşeleri (KLT ile aynı seçim) -> kıyı/yapı gibi kameraya bağlı sabit noktalar.
                Su yüzeyi ve gökyüzü gibi kendi hareketi olan/dokusuz alanlar büyük ölçüde dışarıda kalır.
    "gradient": ızgara üzerinde gradyanı yüksek noktalar (v2 davranışı)."""
    H, W = gray.shape
    if config.RAFT_FIT_POINTS == "corners":
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=6, blockSize=3)
        if pts is not None and len(pts) >= 12:
            pts = pts.reshape(-1, 2)
            return np.clip(np.round(pts[:, 0]), 0, W - 1).astype(int), np.clip(np.round(pts[:, 1]), 0, H - 1).astype(int)
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    gmag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))[ys, xs]
    keep = gmag >= np.quantile(gmag, grad_quantile)
    return xs[keep], ys[keep]


def fit_similarity(flow, frame_bgr, step=8, grad_quantile=0.6, thresh=1.0):
    """
    RAFT akış alanına (2,H,W) RANSAC ile benzerlik dönüşümü (öteleme + dönme + ölçek) oturtur.
    Medyan yalnızca ötelemeyi verir; görüntü merkezi etrafındaki dönmede (tekne yalpası) akış teğetsel olduğu için
    medyan ~0 çıkar. Akış, config.RAFT_FIT_POINTS'e göre seçilen noktalarda örneklenir.
    Dönüş: [tx, ty, dθ, dlog(s)] — öteleme görüntü merkezi etrafında: x' = A(x - c) + c + t
    """
    H, W = flow.shape[1:]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    xs, ys = _sample_points(gray.astype(np.float32) if config.RAFT_FIT_POINTS != "corners" else gray, step,
                            grad_quantile)
    src = np.stack([xs, ys], 1).astype(np.float32)
    dst = src + np.stack([flow[0][ys, xs], flow[1][ys, xs]], 1)
    m = None
    if len(src) >= 8:
        m, _ = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.RANSAC,
                                           ransacReprojThreshold=thresh, maxIters=1000)
    if m is None:
        return np.zeros(4, np.float32)
    c = np.array([W / 2.0, H / 2.0])
    A = m[:, :2]
    t_c = m[:, 2] + A @ c - c
    s = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    return np.array([t_c[0], t_c[1], np.arctan2(m[1, 0], m[0, 0]), np.log(max(s, 1e-6))], np.float32)


class _RaftArgs:
    """RAFT hem `'x' in args` hem `args.x` kullandığı için melez argüman sınıfı (stage3 ile aynı)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

    def __contains__(self, key):
        return key in self.__dict__


class RaftGlobalMotion:
    def __init__(self, device=None, iters=config.RAFT_ITERS, batch_size=None, bf16=False):
        self.device = get_device(device)
        self.iters = iters
        # bf16: yalnızca CPU'da (AMX/AVX512-BF16) ~2x hız; medyan akıştaki fark ~0.003 px
        self.bf16 = bool(bf16) and self.device.type == "cpu"
        args = _RaftArgs(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0)
        model = RAFT(args)
        state = torch.load(config.RAFT_WEIGHTS, map_location="cpu")
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        if batch_size is None:
            batch_size = 1 if self.device.type == "cpu" else 4
        self.batch_size = batch_size

    @staticmethod
    def flow_size(width, height, flow_width=config.FLOW_WIDTH):
        ratio = flow_width / float(width)
        fw = (flow_width // 8) * 8
        fh = (int(height * ratio) // 8) * 8
        return fw, fh

    def _to_tensor(self, frames):
        # BGR -> RGB, (N,H,W,3) -> (N,3,H,W)
        arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)

    @torch.no_grad()
    def _flow_batch(self, prev_frames, next_frames):
        im1 = self._to_tensor(prev_frames)
        im2 = self._to_tensor(next_frames)
        padder = InputPadder(im1.shape)
        im1, im2 = padder.pad(im1, im2)
        if self.bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, flow_up = self.model(im1, im2, iters=self.iters, test_mode=True)
            flow_up = flow_up.float()
        else:
            _, flow_up = self.model(im1, im2, iters=self.iters, test_mode=True)
        flow_up = padder.unpad(flow_up)  # (B,2,H,W)
        b = flow_up.shape[0]
        med = flow_up.reshape(b, 2, -1).median(dim=2).values.cpu().numpy()  # (B,2)
        flows = flow_up.float().cpu().numpy()
        out = []
        for i in range(b):
            out.append(np.concatenate([med[i], fit_similarity(flows[i], prev_frames[i])]))
        return np.asarray(out, np.float32)   # (B,6): med_dx, med_dy, tx, ty, dθ, dlog(s)

    def video_global_shifts(self, video_path, log_every=200):
        """Her ardışık kare çifti için (N-1, 6): medyan dx, dy + RAFT akışına oturtulan benzerlik dönüşümü
        (merkez etrafında tx, ty [px], dθ [rad], dlog(s)). Hepsi FLOW_WIDTH ölçeğinde."""
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw, fh = self.flow_size(W, H)

        ok, prev = cap.read()
        if not ok:
            raise RuntimeError("Video okunamadı: {}".format(video_path))
        prev = cv2.resize(prev, (fw, fh), interpolation=cv2.INTER_AREA)

        shifts = []
        buf_prev, buf_next = [], []
        t0 = time.time()
        n_read = 1
        while True:
            ok, cur = cap.read()
            if not ok:
                break
            n_read += 1
            cur = cv2.resize(cur, (fw, fh), interpolation=cv2.INTER_AREA)
            buf_prev.append(prev)
            buf_next.append(cur)
            prev = cur
            if len(buf_prev) == self.batch_size:
                shifts.append(self._flow_batch(buf_prev, buf_next))
                buf_prev, buf_next = [], []
            if n_read % log_every == 0:
                el = time.time() - t0
                print("      RAFT {}/{} kare  ({:.1f} kare/sn)".format(n_read, total, n_read / max(el, 1e-6)))
        if buf_prev:
            shifts.append(self._flow_batch(buf_prev, buf_next))
        cap.release()

        shifts = np.concatenate(shifts, axis=0).astype(np.float32) if shifts else np.zeros((0, 6), np.float32)
        info = {"width": W, "height": H, "flow_w": fw, "flow_h": fh, "n_frames": n_read}
        return shifts, info
