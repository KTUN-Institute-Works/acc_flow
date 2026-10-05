"""
Benzerlik (öteleme + dönme) düzeltmesini videoya uygulama ve siyah kenarsız ortak kırpma.

Düzeltme her kare için görüntü MERKEZİ etrafında uygulanır:
    x' = R(dθ) (x - c) + c + (dx, dy)
"""
import os

import cv2
import numpy as np


def correction_matrices(corr, W, H):
    """corr: (N, 2..4) -> dx, dy [, dθ (rad) [, dlog(s)]] (orijinal çözünürlük px) -> N adet 2x3 matris."""
    corr = np.asarray(corr, np.float64)
    c = np.array([W / 2.0, H / 2.0])
    mats = []
    for row in corr:
        dx, dy = row[0], row[1]
        da = row[2] if len(row) > 2 else 0.0
        s = np.exp(row[3]) if len(row) > 3 else 1.0
        A = s * np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]])
        t = c - A @ c + np.array([dx, dy])
        mats.append(np.float32(np.hstack([A, t[:, None]])))
    return mats


def common_center_crop(mats, W, H):
    """Tüm kareler için siyah kenar içermeyen, merkezli, en-boy oranı korunan en büyük kırpma oranı k (0..1]."""
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
    k_min = 1.0
    for M in mats:
        if np.allclose(M, [[1, 0, 0], [0, 1, 0]]):
            continue
        quad = cv2.transform(corners, M).reshape(-1, 2)
        lo, hi = 0.2, 1.0
        for _ in range(18):
            k = (lo + hi) / 2
            w2, h2 = W * k / 2, H * k / 2
            pts = [(W / 2 - w2, H / 2 - h2), (W / 2 + w2, H / 2 - h2), (W / 2 + w2, H / 2 + h2), (W / 2 - w2, H / 2 + h2)]
            if all(cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0 for x, y in pts):
                lo = k
            else:
                hi = k
        k_min = min(k_min, lo)
    return k_min


def warp_and_write(video_path, out_path, mats, timer=None):
    """Matrisleri uygular, ortak merkez kırpma yapar ve yazar. Dönüş: (crop (x1,y1,x2,y2), warp_saniye, kare_sayısı)."""
    import time
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = common_center_crop(mats, W, H)
    cw, ch = int(W * k) // 2 * 2, int(H * k) // 2 * 2
    x1, y1 = (W - cw) // 2, (H - ch) // 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    warp_t, n = 0.0, 0
    for M in mats:
        ok, f = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        g = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        g = g[y1:y1 + ch, x1:x1 + cw]
        warp_t += time.perf_counter() - t0
        wr.write(np.ascontiguousarray(g))
        n += 1
    cap.release()
    wr.release()
    return (x1, y1, x1 + cw, y1 + ch), warp_t, n
