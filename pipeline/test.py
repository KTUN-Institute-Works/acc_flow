"""
Test: eğitimde hiç görülmemiş IDA_split/test klipleri üzerinde 4 durum karşılaştırması.

  alpha_0.0  : stabilizasyon yok (alpha = 0)          -> referans / orijinal
  alpha_0.5  : sabit yarım güç
  alpha_1.0  : sabit tam güç
  alpha_mean : kontrol deneyi — sabit alpha = adaptif alpha'nın o klipteki ortalaması
  adaptive   : modelin kare-kare tahmin ettiği alpha (tez yöntemi)

Her durum için: düzeltme = -alpha_t * (dx_t, dy_t, dθ_t) (merkez etrafında öteleme + dönme), tüm video için ortak
kırpma, stabilize video kaydı ve DUTCode/NNDVS metrikleri. Test kliplerinin RAFT ground truth'u varsa
tahmin hatası ve 'kalan sarsıntı' (residual jitter) da raporlanır.

Çıktılar (results/ida_split/test/):
  videos/<clip_id>/<mode>.mp4     -> test klipleri x 4 durum (varsayılan 4 klip = 16 video)
  metrics.csv / metrics.json      -> klip x durum sonuçları (+ RAFT referans satırları)
  summary.csv                     -> durum bazında ortalama
  prediction_accuracy.csv         -> klip bazında dx,dy / alpha tahmin hatası
  report.txt                      -> hangi durum hangi metrikte en iyi
  signals/<clip_id>.npz           -> tahmin ve ground truth sinyalleri (grafikler için)
  plots/en, plots/tr              -> İngilizce ve Türkçe grafikler
"""
import csv
import json
import math
import os
import time

import cv2
import numpy as np

import config
from pipeline import dataset
from pipeline.device import get_device
from pipeline.metrics import dut_metrics
from pipeline.model_io import load_model, predict_sequence
from pipeline.prepare import prepare_clips

# CropArea: stabilize videoda kalan görüntü alanı oranı (bizim ortak kırpmamız). DUTCode CropRatio yalnızca siyah
# kenarları ölçtüğü için siyah kenarsız kırpılmış videolarda hep ~1 çıkar; kırpma maliyetini CropArea gösterir.
METRIC_KEYS = ["Stability", "CropArea", "CropRatio", "MinCropRatio", "Distortion", "ResidualJitter", "FPS"]


def _flow_scale(W, H, d=None):
    """Model biriminden (kısa kenar = FLOW_WIDTH) orijinal piksele ölçek (sx, sy). Hedefler ve tahminler aynı
    birimde olduğundan dikey (IDA) ve yatay (yürüyüş) videolar için aynı formül geçerlidir."""
    u = dataset.model_unit((W, H))
    return u, u


def stabilize_video(video_path, out_path, offsets, alphas, sx, sy, max_ratio=config.MAX_SHIFT_RATIO,
                    max_rot_deg=config.MAX_ROT_DEG):
    """
    offsets: (N,2) dx, dy ya da (N,3) dx, dy, dθ — tahmin edilen jitter (dx, dy model biriminde px, dθ rad)
    alphas : (N,) 0..1 — düzeltmenin gücü (dönme dahil tüm bileşenlere uygulanır)
    Düzeltme = -alpha * jitter, görüntü merkezi etrafında öteleme + dönme; tüm video için ortak merkez kırpma.
    """
    from pipeline.warp import correction_matrices, warp_and_write

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(offsets))
    cap.release()

    a = alphas[:n]
    dx = -(offsets[:n, 0] * sx * a)
    dy = -(offsets[:n, 1] * sy * a)
    dth = -(offsets[:n, 2] * a) if offsets.shape[1] > 2 else np.zeros(n)
    lim_x, lim_y, lim_r = max_ratio * W, max_ratio * H, np.radians(max_rot_deg)
    clipped = int(np.sum((np.abs(dx) > lim_x) | (np.abs(dy) > lim_y) | (np.abs(dth) > lim_r)))
    dx, dy, dth = np.clip(dx, -lim_x, lim_x), np.clip(dy, -lim_y, lim_y), np.clip(dth, -lim_r, lim_r)

    mats = correction_matrices(np.stack([dx, dy, dth], 1), W, H)
    crop, warp_t, written = warp_and_write(video_path, out_path, mats)
    return {"crop": crop, "warp_seconds": warp_t, "n_frames": written,
            "clipped_frames": clipped, "dx": dx, "dy": dy, "dtheta": dth}


def _corr(a, b):
    """Pearson korelasyonu; sabit sinyalde (örn. tahmin edilmeyen dx = 0) NaN."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _jitter_rms(jit, rot_scale):
    """(N,D) jitter -> tek sayı RMS (px). dθ (rad), rot_scale (yarım köşegen px) ile köşe kaymasına çevrilir."""
    w = np.ones(jit.shape[1])
    if jit.shape[1] > 2:
        w[2] = rot_scale
    return float(np.sqrt(np.mean(np.sum((jit * w) ** 2, 1))))


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def evaluate_mode(clip, cid, mode, offsets, alphas, sx, sy, d, out_video, infer_t, rot_scale=0.0):
    """Bir durumu (offsets * alphas) uygular, videoyu yazar, DUTCode metriklerini hesaplar ve satır döndürür.
    d: ground truth içeren önbellek (yoksa None). infer_t: model çıkarım süresi (RAFT referansı için None)."""
    st = stabilize_video(clip["video"], out_video, offsets, alphas, sx, sy)
    fps_proc = st["n_frames"] / (infer_t + st["warp_seconds"]) if infer_t is not None else float("nan")
    print("    {:<14s} -> kırpma {}x{}  | metrikler hesaplanıyor...".format(
        mode, st["crop"][2] - st["crop"][0], st["crop"][3] - st["crop"][1]))
    m = dut_metrics(clip["video"], out_video)
    row = {"clip_id": cid, "domain": dataset.domain_of(cid), "mode": mode, "alpha": None,
           "mean_alpha_used": float(np.mean(alphas))}
    row.update(m)
    if d is not None:
        n = min(len(offsets), len(d["jitter"]))
        resid = d["jitter"][:n] - alphas[:n, None] * offsets[:n, :d["jitter"].shape[1]]
        row["ResidualJitter"] = _jitter_rms(resid, rot_scale)
    else:
        row["ResidualJitter"] = float("nan")
    row["FPS"] = float(fps_proc)
    row["crop_w"] = st["crop"][2] - st["crop"][0]
    row["crop_h"] = st["crop"][3] - st["crop"][1]
    info = dataset.video_info(clip["video"])
    row["CropArea"] = float(row["crop_w"] * row["crop_h"]) / float(info["width"] * info["height"])
    row["clipped_frames"] = st["clipped_frames"]
    row["video"] = os.path.relpath(out_video, config.PROJECT_ROOT)
    print("                   Stability {:.3f} | CropArea {:.3f} | Distortion {:.3f} | Residual {:.3f} px".format(
        m["Stability"], row["CropArea"], m["Distortion"], row["ResidualJitter"]))
    return row


def _comparison_video(original, panels, out_path):
    """Orijinal + stabilize videoları yan yana koyar (her panel orijinal boyuta ölçeklenir, etiketli)."""
    caps = [cv2.VideoCapture(original)] + [cv2.VideoCapture(p) for p, _ in panels]
    labels = ["Original"] + [lab for _, lab in panels]
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    sep = 6
    out_w = W * len(caps) + sep * (len(caps) - 1)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, H))
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                frames = None
                break
            frames.append(cv2.resize(f, (W, H)))
        if frames is None:
            break
        row = []
        for i, (f, lab) in enumerate(zip(frames, labels)):
            cv2.putText(f, lab, (14, 38), font, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(f, lab, (14, 38), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(f)
            if i < len(frames) - 1:
                row.append(np.full((H, sep, 3), 255, np.uint8))
        writer.write(np.hstack(row))
    for c in caps:
        c.release()
    writer.release()


def run_test(clip_ids=None, all_test=False, model_path=None, device=None, raft_bf16=False, raft_reference="compute",
             out_dir=None):
    """
    Stabilizasyon YALNIZCA IMU ile yapılır (model: IMU -> dx, dy, dθ, α). Test sırasında RAFT çalıştırılmaz.

    raft_reference: RAFT yalnızca değerlendirme/karşılaştırma içindir (tahmin doğruluğu, kalan sarsıntı, RAFT üst sınırı)
        "cache"   (varsayılan) -> klibin RAFT önbelleği daha önce oluşmuşsa karşılaştırmaya eklenir; yoksa atlanır.
        "off"     -> RAFT karşılaştırması hiç yapılmaz.
        "compute" (varsayılan) -> RAFT yalnızca önbelleği olmayan kliplerde BİR KEZ çalışır (yeni araba/vapur test
                    klipleri, ~15 sn); sonraki testlerde önbellekten okunur.
    """
    model_path = model_path or config.MODEL_PATH
    out_dir = out_dir or config.TEST_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("ACCFlow | TEST (görülmemiş klipler: IDA_split/test + araba/vapur)")
    print("=" * 70)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model bulunamadı: {}  ->  önce `python main.py train` çalıştırın.".format(model_path))

    clips = dataset.load_clips("test") if all_test else dataset.select_test_clips(clip_ids)
    train_ids = {c["clip_id"] for c in dataset.load_clips("train")}
    for c in clips:
        assert c["clip_id"] not in train_ids, "Test klibi eğitim setinde! {}".format(c["clip_id"])
    print("Test klipleri: {}".format([c["clip_id"] for c in clips]))

    dev = get_device(device)
    model, mean, std, meta = load_model(model_path, dev)
    print("Model: {} (epoch {})  | cihaz: {}".format(os.path.relpath(model_path, config.PROJECT_ROOT),
                                                    meta.get("epoch", "?"), dev))

    # 1) Model girdisi: yalnızca IMU özellikleri (RAFT yok)
    data = prepare_clips(clips, need_gt=False)
    # 2) Değerlendirme için RAFT referansı (isteğe bağlı; varsayılan: yalnızca önbellekte varsa)
    if raft_reference == "compute":
        data.update(prepare_clips(clips, need_gt=True, device=device, bf16=raft_bf16))
    elif raft_reference == "cache":
        cached = [c for c in clips if dataset.has_gt_cache(c)]
        if cached:
            data.update({c["clip_id"]: dataset.prepare_clip(c, need_gt=True) for c in cached})
        missing = [c["clip_id"] for c in clips if not dataset.has_gt_cache(c)]
        if missing:
            print("RAFT referansı önbellekte yok (bu kliplerde yalnızca IMU sonuçları + DUTCode metrikleri): {}".format(missing))
    print("Stabilizasyon yalnızca IMU ile yapılıyor; RAFT karşılaştırması: {}".format(raft_reference))

    rows, acc_rows = [], []
    for ci, clip in enumerate(clips):
        cid = clip["clip_id"]
        d = data[cid]
        info = dataset.video_info(clip["video"])
        W, H = info["width"], info["height"]
        sx, sy = _flow_scale(W, H, d)
        print("\n[{}/{}] {}  ({}x{}, {} kare)".format(ci + 1, len(clips), cid, W, H, info["n_frames"]))

        # --- Model çıkarımı (yalnızca IMU) ---
        t0 = time.perf_counter()
        offsets, alpha_pred = predict_sequence(model, d["imu_jitter"], mean, std, dev,
                                               target_std=np.asarray(meta.get("target_std", 1.0), np.float32))
        rot_scale = dataset.half_diag(d)
        # Model yalnızca bazı bileşenleri tahmin edebilir (örn. dy, dθ). Tahmin edilmeyenler (dx) 0 = düzeltilmez.
        comps = meta.get("motion_components") or ["dx", "dy", "dtheta"][:offsets.shape[1]]
        full_dim = 3 if config.TARGET != "median" else 2
        full = np.zeros((len(offsets), full_dim), np.float32)
        full[:, dataset.motion_index(comps)] = offsets
        offsets = full
        infer_t = time.perf_counter() - t0

        has_gt = "jitter" in d
        if has_gt:
            n = min(len(offsets), len(d["jitter"]))
            gt_j = d["jitter"][:n]
            if config.ALPHA_MODE == "optimal":
                # alpha hedefi: bu modelin tahminine göre ideal düzeltme oranı (RAFT'tan; yalnızca değerlendirme)
                gt_a = dataset.optimal_alpha(gt_j, offsets[:n], rot_scale)
            else:
                gt_a = d["alpha"][:n]
            err = offsets[:n] - gt_j
            acc = {
                "clip_id": cid,
                "RMSE_dx_px": float(np.sqrt(np.mean(err[:, 0] ** 2))),
                "RMSE_dy_px": float(np.sqrt(np.mean(err[:, 1] ** 2))),
                "GT_jitter_RMS_px": _jitter_rms(gt_j, rot_scale),
                "Corr_dx": _corr(offsets[:n, 0], gt_j[:, 0]),
                "Corr_dy": _corr(offsets[:n, 1], gt_j[:, 1]),
            }
            if gt_j.shape[1] > 2:
                acc["RMSE_dtheta_deg"] = float(np.degrees(np.sqrt(np.mean(err[:, 2] ** 2))))
                acc["GT_dtheta_RMS_deg"] = float(np.degrees(np.sqrt(np.mean(gt_j[:, 2] ** 2))))
                acc["Corr_dtheta"] = _corr(offsets[:n, 2], gt_j[:, 2])
            acc.update({
                "MAE_alpha": float(np.mean(np.abs(alpha_pred[:n] - gt_a))),
                "Corr_alpha": float(np.corrcoef(alpha_pred[:n], gt_a)[0, 1]) if np.std(gt_a) > 1e-6 else float("nan"),
                "pan_frac_gt": float(np.mean(gt_a < 0.5)),      # alpha hedefinin 0.5'in altında olduğu kare oranı
                "mean_alpha_pred": float(np.mean(alpha_pred[:n])),
                "mean_alpha_gt": float(np.mean(gt_a)),
            })
            acc_rows.append(acc)
            msg = "    Tahmin: corr dx {:+.2f} dy {:+.2f}".format(acc["Corr_dx"], acc["Corr_dy"])
            if "Corr_dtheta" in acc:
                msg += " dθ {:+.2f} (RMSE {:.3f}°, GT RMS {:.3f}°)".format(
                    acc["Corr_dtheta"], acc["RMSE_dtheta_deg"], acc["GT_dtheta_RMS_deg"])
            msg += " | alpha MAE {:.3f} (ort. tahmin {:.2f}, GT {:.2f})".format(
                acc["MAE_alpha"], acc["mean_alpha_pred"], acc["mean_alpha_gt"])
            print(msg)

        sig_dir = os.path.join(out_dir, "signals")
        os.makedirs(sig_dir, exist_ok=True)
        sig = {"pred_offsets": offsets, "pred_alpha": alpha_pred, "fps": info["fps"]}
        if has_gt:
            sig.update({"gt_jitter": d["jitter"], "gt_alpha": gt_a, "gt_trajectory": d["trajectory"],
                        "gt_smooth": d["smooth"]})
        np.savez_compressed(os.path.join(sig_dir, cid + ".npz"), **sig)

        for mode, const_alpha in config.MODES:
            if const_alpha is None:                       # adaptif (tez)
                alphas = alpha_pred
            elif const_alpha == "mean":                   # kontrol: adaptifin ortalamasına eşit sabit alpha
                alphas = np.full(len(offsets), float(np.mean(alpha_pred)), np.float32)
            else:
                alphas = np.full(len(offsets), float(const_alpha), np.float32)
            row = evaluate_mode(clip, cid, mode, offsets, alphas, sx, sy, d if has_gt else None,
                                os.path.join(out_dir, "videos", cid, mode + ".mp4"), infer_t, rot_scale)
            row["alpha"] = "adaptive" if const_alpha is None else const_alpha
            row["std_alpha_used"] = float(np.std(alphas))
            rows.append(row)

        # --- RAFT referansı: modelin öğrenmeye çalıştığı hedefin kendisiyle stabilizasyon (üst sınır) ---
        if has_gt:
            n = min(len(offsets), len(d["jitter"]))
            for mode, ref_alpha in config.REFERENCE_MODES:
                ref_off = d["jitter"][:n]
                if ref_alpha == "opt":      # IMU tahmini + ideal alpha* (alpha başının üst sınırı)
                    ref_off = offsets[:n]
                if ref_alpha == "obs":      # yalnızca modelin düzelttiği bileşenler (dx yok)
                    mask = np.zeros(ref_off.shape[1], np.float32)
                    mask[[i for i in dataset.motion_index(comps) if i < ref_off.shape[1]]] = 1.0
                    ref_off = ref_off * mask
                if ref_alpha == "gt":
                    alphas = d["alpha"][:n]
                elif ref_alpha == "opt":
                    alphas = gt_a[:n]
                else:
                    alphas = np.full(n, 1.0 if ref_alpha == "obs" else float(ref_alpha), np.float32)
                row = evaluate_mode(clip, cid, mode, ref_off, alphas, sx, sy, d,
                                    os.path.join(out_dir, "videos", cid, mode + ".mp4"), None, rot_scale)
                row["alpha"] = {"gt": "raft_gt", "obs": 1.0, "opt": "alpha_opt"}.get(ref_alpha, ref_alpha)
                rows.append(row)

        # Yan yana karşılaştırma videosu: orijinal | IMU α=1 | IMU adaptif | (varsa) RAFT α=1
        panels = [(os.path.join(out_dir, "videos", cid, m + ".mp4"), lab) for m, lab in
                  [("alpha_1.0", "IMU (a=1)"), ("adaptive", "IMU (adaptive)"), ("raft_gt", "RAFT (a=1)")]]
        _comparison_video(clip["video"], [p for p in panels if os.path.exists(p[0])],
                          os.path.join(out_dir, "videos", cid, "compare.mp4"))

    # --- Kaydet ---
    _write_csv(os.path.join(out_dir, "metrics.csv"), rows)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if acc_rows:
        _write_csv(os.path.join(out_dir, "prediction_accuracy.csv"), acc_rows)

    summary = []
    for mode, _ in list(config.MODES) + list(config.REFERENCE_MODES):
        mr = [r for r in rows if r["mode"] == mode]
        s = {"mode": mode}
        for k in METRIC_KEYS:
            s[k] = float(np.nanmean([r[k] for r in mr])) if mr else float("nan")
        summary.append(s)
    _write_csv(os.path.join(out_dir, "summary.csv"), summary)

    report = _make_report(rows, summary, acc_rows, meta, model_path)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)

    from pipeline.plots import make_all_plots
    make_all_plots(out_dir)
    print("Sonuçlar: {}".format(out_dir))
    return rows


def _sign_test_p(wins, losses):
    """Tek yönlü işaret testi (eşitlikler hariç): P(X >= wins), X ~ Binom(wins+losses, 0.5)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2.0 ** n


def _paired_comparison(rows, mode_a, mode_b, title, better_high, tol=1e-4):
    """Klip bazında eşleştirilmiş karşılaştırma: mode_a vs mode_b, kazanç/eşit/kayıp sayıları ve işaret testi."""
    clips = sorted({r["clip_id"] for r in rows})
    by = {(r["clip_id"], r["mode"]): r for r in rows}
    pairs = [(c, by[(c, mode_a)], by[(c, mode_b)]) for c in clips if (c, mode_a) in by and (c, mode_b) in by]
    if not pairs:
        return []
    keys = [k for k in better_high if not all(np.isnan(p[1][k]) for p in pairs)]
    lines = ["TEZ KARŞILAŞTIRMASI: {} vs {} (klip bazında, a/b):".format(mode_a, title)]
    lines.append("  {:<30s} {:>7s} {:>7s}".format("clip", "ᾱ", "σ(α)") +
                 "".join(" {:>21s}".format(k) for k in keys))
    for c, ra, rb in pairs:
        line = "  {:<30s} {:>7.3f} {:>7.3f}".format(c, ra["mean_alpha_used"], ra.get("std_alpha_used", float("nan")))
        for k in keys:
            line += " {:>10.4f}/{:<10.4f}".format(ra[k], rb[k])
        lines.append(line)
    lines.append("  Özet / summary ({} daha iyi / eşit / daha kötü, tek yönlü işaret testi p; |fark| < {} = eşit):".format(
        mode_a, tol))
    for k in keys:
        w = t = l = 0
        diffs = []
        for _, ra, rb in pairs:
            if np.isnan(ra[k]) or np.isnan(rb[k]):
                continue
            diff = (ra[k] - rb[k]) if better_high[k] else (rb[k] - ra[k])   # >0 = mode_a daha iyi
            diffs.append(diff)
            if abs(diff) < tol:
                t += 1
            elif diff > 0:
                w += 1
            else:
                l += 1
        lines.append("    {:<15s} {}/{}/{}  p={:.3f}  ortalama fark (iyi yönde) {:+.4f}".format(
            k, w, t, l, _sign_test_p(w, l), float(np.mean(diffs)) if diffs else float("nan")))
    lines.append("")
    return lines


def _make_report(rows, summary, acc_rows, meta, model_path):
    better_high = {"Stability": True, "CropArea": True, "Distortion": True, "ResidualJitter": False}
    model_modes = {m for m, _ in config.MODES}
    ref_summary = [s for s in summary if s["mode"] not in model_modes]
    summary = [s for s in summary if s["mode"] in model_modes]
    lines = ["ACCFlow test raporu / test report",
             "Model: {} (epoch {})".format(os.path.relpath(model_path, config.PROJECT_ROOT), meta.get("epoch", "?")),
             "Klipler / clips: {}".format(sorted({r["clip_id"] for r in rows})), "",
             "Durum bazında ortalamalar / mean per mode:"]
    hdr = "{:<15s}".format("mode") + "".join("{:>15s}".format(k) for k in METRIC_KEYS)
    lines.append(hdr)
    for s in summary:
        lines.append("{:<15s}".format(s["mode"]) + "".join("{:>15.4f}".format(s[k]) for k in METRIC_KEYS))
    if ref_summary:
        lines.append("RAFT referansı (üst sınır; model değil) / RAFT reference (upper bound, not the model):")
        for s in ref_summary:
            lines.append("{:<15s}".format(s["mode"]) + "".join("{:>15.4f}".format(s[k]) for k in METRIC_KEYS))
    lines.append("")
    lines.append("En iyi durum (ortalama) / best mode (mean):")
    for k, hi in better_high.items():
        vals = [(s[k], s["mode"]) for s in summary if not np.isnan(s[k])]
        if not vals:
            continue
        v, m = (max if hi else min)(vals)
        lines.append("  {:<15s} -> {} ({:.4f}) [{}]".format(k, m, v, "yüksek iyi" if hi else "düşük iyi"))
    # Stabilizasyon yapan durumlar arasında (alpha=0 hariç) en iyisi
    lines.append("")
    lines.append("Stabilizasyon uygulayan durumlar arasında (alpha_0.0 hariç) / among stabilizing modes:")
    for k, hi in better_high.items():
        vals = [(s[k], s["mode"]) for s in summary if s["mode"] != "alpha_0.0" and not np.isnan(s[k])]
        if vals:
            v, m = (max if hi else min)(vals)
            lines.append("  {:<15s} -> {} ({:.4f})".format(k, m, v))
    lines.append("")
    for ref, ref_name in (("alpha_1.0", "sabit α=1 / fixed α=1"),
                          ("alpha_mean", "sabit α=ort(adaptif) / fixed α=mean(adaptive)  [kontrol deneyi]")):
        lines += _paired_comparison(rows, "adaptive", ref, ref_name, better_high)
    domains = sorted({dataset.domain_of(r["clip_id"]) for r in rows})
    if len(domains) > 1:
        lines.append("Alan bazında ortalamalar / mean per domain (IDA = hafif sarsıntı; araba/vapur = belirgin sarsıntı):")
        keys = ["Stability", "CropArea", "Distortion", "ResidualJitter"]
        for dom in domains:
            dr = [r for r in rows if dataset.domain_of(r["clip_id"]) == dom]
            lines.append("  [{}] {} klip".format(dom, len({r["clip_id"] for r in dr})))
            lines.append("  {:<15s}".format("mode") + "".join("{:>15s}".format(k) for k in keys))
            for mode in [m for m, _ in list(config.MODES) + list(config.REFERENCE_MODES)]:
                mr = [r for r in dr if r["mode"] == mode]
                if mr:
                    lines.append("  {:<15s}".format(mode) + "".join(
                        "{:>15.4f}".format(float(np.nanmean([r[k] for r in mr]))) for k in keys))
        lines.append("")
    lines.append("Klip bazında Stability kazananı / per-clip Stability winner:")
    for cid in sorted({r["clip_id"] for r in rows}):
        cr = [r for r in rows if r["clip_id"] == cid and r["mode"] in model_modes]
        best = max(cr, key=lambda r: r["Stability"])
        lines.append("  {:<30s} -> {} ({:.4f})".format(cid, best["mode"], best["Stability"]))
    if acc_rows:
        lines.append("")
        lines.append("Tahmin doğruluğu / prediction accuracy (RAFT GT'ye göre):")
        for a in acc_rows:
            line = "  {:<30s} corr dx {:+.2f} dy {:+.2f}".format(a["clip_id"], a["Corr_dx"], a["Corr_dy"])
            if "Corr_dtheta" in a:
                line += " dθ {:+.2f}".format(a["Corr_dtheta"])
            line += " | RMSE dx {:.3f} dy {:.3f} px | alpha MAE {:.3f} corr {:+.2f} | pan karesi (GT) %{:.0f}".format(
                a["RMSE_dx_px"], a["RMSE_dy_px"], a["MAE_alpha"], a["Corr_alpha"], 100 * a["pan_frac_gt"])
            lines.append(line)
    lines.append("")
    lines.append("Not: Stability/CropRatio/Distortion DUTCode-NNDVS MetricAnalyzer ile hesaplanmıştır (yüksek = iyi).")
    lines.append("ResidualJitter: RAFT hedefinden düzeltme sonrası kalan sarsıntının RMS'i (px, {}px genişlik ölçeği; dönme köşedeki kaymaya çevrilmiştir; düşük = iyi).".format(config.FLOW_WIDTH))
    lines.append("CropArea: kırpma sonrası kalan alan oranı (yüksek = iyi). Adaptif α'nın beklenen faydası burada görünür.")
    if config.ALPHA_MODE == "optimal":
        lines.append("α hedefi: optimal (α* = modelin tahminine göre kalan sarsıntıyı en aza indiren düzeltme oranı, "
                     "RAFT'tan; testte yalnızca değerlendirme için). 'pan karesi' sütunu = α* < 0.5 kare oranı.")
        lines.append("imu_alpha_opt: aynı IMU tahmini + ideal α* -> α başının ulaşabileceği üst sınır.")
    else:
        lines.append("α hedefi: {} (v0={} px/kare). Pan karesi = α_gt < 0.5.".format(config.ALPHA_MODE, config.ALPHA_PAN_V0))
    lines.append("FPS: yalnızca IMU modeli + warp süresi (video okuma/yazma ve RAFT hariç).")
    return "\n".join(lines) + "\n"
