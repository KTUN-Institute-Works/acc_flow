"""
RAFT referans videoları: modelin eğitim hedefi (RAFT'tan çıkarılan dx, dy, alpha) videoya uygulanınca
görüntü gerçekte ne kadar stabilize oluyor? Eğitim verisinin doğru hazırlandığını gözle kontrol etmek için.

Her klip için (results/ida_split/raft_reference/<clip_id>/):
  raft_alpha1.mp4   : RAFT jitter'ı (öteleme + dönme) tam güçle (alpha = 1) geri alınmış video
  raft_alphaGT.mp4  : RAFT jitter'ı RAFT'tan hesaplanan alpha_gt ile geri alınmış video (modelin hedefi)
  compare.mp4       : Orijinal | RAFT (a=1) | RAFT (a=GT) yan yana
  metrics.csv       : (tüm klipler) DUTCode/NNDVS metrikleri, orijinal video dahil
"""
import csv
import os

import numpy as np

import config
from pipeline import dataset
from pipeline.metrics import dut_metrics
from pipeline.prepare import prepare_clips
from pipeline.test import _comparison_video, _flow_scale, stabilize_video


def render_raft_reference(clip_ids=None, split="test", device=None, raft_bf16=False):
    if clip_ids:
        allc = {c["clip_id"]: c for c in dataset.load_clips()}
        clips = [allc[c] for c in clip_ids]
    elif split == "test":
        clips = dataset.select_test_clips()
    else:
        clips = dataset.load_clips(split if split != "all" else None)

    print("=" * 70)
    print("ACCFlow | RAFT REFERANS VİDEOLARI ({} klip)".format(len(clips)))
    print("=" * 70)
    data = prepare_clips(clips, need_gt=True, device=device, bf16=raft_bf16)
    out_root = config.RAFT_REFERENCE_DIR
    rows = []
    for clip in clips:
        cid = clip["clip_id"]
        d = data[cid]
        info = dataset.video_info(clip["video"])
        sx, sy = _flow_scale(info["width"], info["height"], d)
        n = len(d["jitter"])
        out_dir = os.path.join(out_root, cid)
        os.makedirs(out_dir, exist_ok=True)
        jit = d["jitter"]
        jit_rms = float(np.sqrt(np.mean(np.sum(jit[:, :2] ** 2, 1))))
        msg = "\n{}  | hedef: {} | öteleme jitter RMS {:.2f} px (orijinal çözünürlük)".format(
            cid, config.TARGET, jit_rms * sx)
        if jit.shape[1] > 2:
            rot = float(np.sqrt(np.mean(jit[:, 2] ** 2)))
            msg += " | dönme jitter RMS {:.3f}° (köşede {:.2f} px)".format(
                np.degrees(rot), rot * dataset.half_diag(d) * sx)
        print(msg + " | ort. alpha_gt {:.2f}".format(float(np.mean(d["alpha"]))))

        variants = [("original", None), ("raft_alpha1", np.ones(n, np.float32)), ("raft_alphaGT", d["alpha"][:n])]
        for name, alphas in variants:
            if alphas is None:
                path = clip["video"]
            else:
                path = os.path.join(out_dir, name + ".mp4")
                st = stabilize_video(clip["video"], path, d["jitter"][:n], alphas, sx, sy)
                print("   {:<13s} kırpma {}x{}".format(name, st["crop"][2] - st["crop"][0], st["crop"][3] - st["crop"][1]))
            m = dut_metrics(clip["video"], path)
            row = {"clip_id": cid, "variant": name, "jitter_rms_px_fullres": jit_rms * sx}
            row.update(m)
            rows.append(row)
            print("   {:<13s} Stability {:.3f} | CropRatio {:.3f} | Distortion {:.3f}".format(
                name, m["Stability"], m["CropRatio"], m["Distortion"]))
        _comparison_video(clip["video"], [
            (os.path.join(out_dir, "raft_alpha1.mp4"), "RAFT (a=1)"),
            (os.path.join(out_dir, "raft_alphaGT.mp4"), "RAFT (a=GT)"),
        ], os.path.join(out_dir, "compare.mp4"))

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nRAFT referans videoları: {}".format(out_root))
    return rows
