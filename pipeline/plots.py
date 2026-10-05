"""
Test ve eğitim sonuçlarının İngilizce ve Türkçe grafikleri (matplotlib).

Üretilen dosyalar (results/ida_split/test/plots/{en,tr}/ ve results/ida_split/training/plots/{en,tr}/):
  metrics_by_clip        : 4 metrik x (test klipleri + ortalama) x 5 durum, gruplu çubuk
  summary_by_mode        : durum bazında ortalama (+ klip noktaları) — "hangisi daha iyi" özeti
  signals_<clip>         : tahmin edilen dx, dy, alpha vs RAFT ground truth
  trajectory_<clip>      : orijinal ve durumlara göre düzeltilmiş kamera yolu
  training_loss          : eğitim / doğrulama kayıpları
"""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Görsel stil (renkler CVD-güvenli olarak doğrulandı: bitişik çiftlerde ΔE >= 9.2)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GT_COLOR = "#52514e"
MODE_COLORS = {
    "alpha_0.0": "#eb6834",   # turuncu
    "alpha_0.5": "#1baf7a",   # su yeşili
    "alpha_1.0": "#4a3aa7",   # mor
    "alpha_mean": "#e87ba4",  # pembe: kontrol deneyi (sabit α = adaptifin ortalaması)
    "adaptive": "#2a78d6",    # mavi (tez yöntemi)
    "raft_gt": "#52514e",     # koyu gri: RAFT referansı (üst sınır)
    "raft_gt_alpha": "#8a8984",
    "raft_gt_obs": "#a9a8a2",     # açık gri: yalnızca dy, dθ düzeltilen RAFT üst sınırı
    "imu_alpha_opt": "#8a8984",   # orta gri: IMU tahmini + ideal α* (α başının üst sınırı)
}
PRED_COLOR = "#2a78d6"

TEXT = {
    "en": {
        "modes": {"alpha_0.0": "α = 0 (no stabilization)", "alpha_0.5": "α = 0.5 (fixed)",
                  "alpha_1.0": "α = 1 (fixed)", "alpha_mean": "Fixed α = mean of adaptive (control)",
                  "adaptive": "Adaptive α (proposed)",
                  "raft_gt": "RAFT reference (α = 1)", "raft_gt_alpha": "RAFT reference (α = GT)",
                  "raft_gt_obs": "RAFT reference (dy, dθ only; α = 1)",
                  "imu_alpha_opt": "IMU prediction + ideal α* (upper bound)"},
        "metrics": {"Stability": "Stability score ↑", "CropRatio": "Crop ratio ↑",
                    "CropArea": "Remaining area after crop ↑",
                    "MinCropRatio": "Min. crop ratio ↑", "Distortion": "Distortion value ↑ (1 = none)",
                    "ResidualJitter": "Residual jitter (px) ↓", "FPS": "Throughput (FPS) ↑"},
        "mean": "Mean",
        "clip": "Test clip",
        "by_clip_title": "Stabilization metrics on unseen test clips — fixed vs. adaptive α",
        "summary_title": "Mean over {n} unseen test clips (small dots = individual clips, shaded = best stabilizing mode)",
        "signals_title": "IMU-only prediction vs. RAFT ground truth — {clip}",
        "dx": "dx jitter (px)", "dy": "dy jitter (px)", "dtheta": "dθ jitter (°)", "alpha": "α",
        "traj_th": "Rotation path (°)",
        "time": "Time (s)", "gt": "RAFT ground truth", "pred": "IMU-Net prediction",
        "traj_title": "Camera path before/after correction — {clip}",
        "traj_x": "Horizontal path (px)", "traj_y": "Vertical path (px)",
        "orig": "Original path", "smooth": "Smooth target",
        "loss_title": "Training history", "epoch": "Epoch", "loss": "MSE loss",
        "total": "Total", "motion": "Motion ({comps})", "alpha_loss": "Alpha",
        "best": "best epoch", "naive": "naive baseline", "loss_norm": "MSE (normalized)",
        "short": {"alpha_0.0": "α = 0", "alpha_0.5": "α = 0.5", "alpha_1.0": "α = 1", "alpha_mean": "α = mean",
                  "adaptive": "Adaptive α",
                  "raft_gt": "RAFT α=1", "raft_gt_alpha": "RAFT α=GT", "raft_gt_obs": "RAFT dy,dθ", "imu_alpha_opt": "IMU + α*"},
        "train": "train", "val": "validation",
    },
    "tr": {
        "modes": {"alpha_0.0": "α = 0 (stabilizasyon yok)", "alpha_0.5": "α = 0,5 (sabit)",
                  "alpha_1.0": "α = 1 (sabit)", "alpha_mean": "Sabit α = adaptif ortalaması (kontrol)",
                  "adaptive": "Adaptif α (önerilen)",
                  "raft_gt": "RAFT referansı (α = 1)", "raft_gt_alpha": "RAFT referansı (α = GT)",
                  "raft_gt_obs": "RAFT referansı (yalnız dy, dθ; α = 1)",
                  "imu_alpha_opt": "IMU tahmini + ideal α* (üst sınır)"},
        "metrics": {"Stability": "Kararlılık skoru ↑", "CropRatio": "Kırpma oranı ↑",
                    "CropArea": "Kırpma sonrası kalan alan ↑",
                    "MinCropRatio": "Min. kırpma oranı ↑", "Distortion": "Bozulma değeri ↑ (1 = yok)",
                    "ResidualJitter": "Kalan sarsıntı (px) ↓", "FPS": "İşleme hızı (FPS) ↑"},
        "mean": "Ortalama",
        "clip": "Test klibi",
        "by_clip_title": "Görülmemiş test kliplerinde stabilizasyon metrikleri — sabit ve adaptif α",
        "summary_title": "{n} görülmemiş test klibinin ortalaması (küçük noktalar = klipler, gölgeli = en iyi stabilize eden durum)",
        "signals_title": "Yalnızca IMU ile tahmin ve RAFT referansı — {clip}",
        "dx": "dx sarsıntı (px)", "dy": "dy sarsıntı (px)", "dtheta": "dθ sarsıntı (°)", "alpha": "α",
        "traj_th": "Dönme yolu (°)",
        "time": "Zaman (sn)", "gt": "RAFT referansı", "pred": "IMU-Net tahmini",
        "traj_title": "Düzeltme öncesi/sonrası kamera yolu — {clip}",
        "traj_x": "Yatay yol (px)", "traj_y": "Dikey yol (px)",
        "orig": "Orijinal yol", "smooth": "Yumuşak hedef",
        "loss_title": "Eğitim geçmişi", "epoch": "Epoch", "loss": "MSE kaybı",
        "total": "Toplam", "motion": "Hareket ({comps})", "alpha_loss": "Alpha",
        "best": "en iyi epoch", "naive": "naif referans", "loss_norm": "MSE (normalize)",
        "short": {"alpha_0.0": "α = 0", "alpha_0.5": "α = 0,5", "alpha_1.0": "α = 1", "alpha_mean": "α = ort.",
                  "adaptive": "Adaptif α",
                  "raft_gt": "RAFT α=1", "raft_gt_alpha": "RAFT α=GT", "raft_gt_obs": "RAFT dy,dθ", "imu_alpha_opt": "IMU + α*"},
        "train": "eğitim", "val": "doğrulama",
    },
}

PLOT_METRICS = ["Stability", "CropArea", "Distortion", "ResidualJitter"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _comps_label():
    from pipeline import dataset
    return ", ".join({"dx": "dx", "dy": "dy", "dtheta": "dθ"}[c] for c in dataset.motion_components())


def _short(cid):
    s = cid.replace("video_", "")
    for p in ("araba_", "vapur_", "walk_"):
        if s.startswith(p):
            return p[:-1] + "\n" + s[len(p):]
    return s.replace("_part", "\npart ")


def _read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return rows


# ---------------------------------------------------------------------------
# Test grafikleri
# ---------------------------------------------------------------------------
# 1'e yakın, dar aralıkta toplanan metrikler (bar yerine yakınlaştırılmış nokta grafiği)
BOUNDED = {"Stability", "CropRatio", "MinCropRatio", "Distortion", "CropArea"}
MODE_MARKERS = {"alpha_0.0": "o", "alpha_0.5": "s", "alpha_1.0": "^", "alpha_mean": "v", "adaptive": "D",
                "raft_gt": "*", "raft_gt_alpha": "X", "raft_gt_obs": "P", "imu_alpha_opt": "h"}


# "En iyi stabilize eden durum" adayları (α = 0 ve RAFT referansları hariç)
STABILIZING_MODES = ("alpha_0.5", "alpha_1.0", "alpha_mean", "adaptive")


def _modes_in(rows):
    """Karşılaştırılan durumlar (config.MODES) + (varsa) RAFT referans durumları."""
    present = {r["mode"] for r in rows}
    return [m for m, _ in config.MODES] + [m for m, _ in getattr(config, "REFERENCE_MODES", []) if m in present]


def _mode_values(rows, metric, mode, groups):
    vals = []
    for g in groups:
        if g == "__mean__":
            v = [r[metric] for r in rows if r["mode"] == mode]
            vals.append(float(np.nanmean(v)) if v else np.nan)
        else:
            v = [r[metric] for r in rows if r["mode"] == mode and r["clip_id"] == g]
            vals.append(v[0] if v else np.nan)
    return vals


def _zoom_ylim(ax, values, upper_cap=None):
    f = [v for v in values if np.isfinite(v)]
    if not f:
        return
    lo, hi = min(f), max(f)
    pad = max((hi - lo) * 0.25, 0.005)
    top = hi + pad * 1.6
    if upper_cap is not None:
        top = min(top, upper_cap + pad)
    ax.set_ylim(lo - pad, top)


def plot_metrics_by_clip(rows, lang, out_dir):
    """Her metrik için: x = test klipleri + ortalama, her grupta 4 durum yan yana."""
    T = TEXT[lang]
    modes = _modes_in(rows)
    clips = sorted({r["clip_id"] for r in rows})
    groups = clips + ["__mean__"]
    fig, axes = plt.subplots(2, 2, figsize=(max(13, 2.3 * (len(groups) + 1)), 8.5))
    width = 0.8 / len(modes)
    x = np.arange(len(groups))
    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        vals_all = []
        for mi, mode in enumerate(modes):
            vals = _mode_values(rows, metric, mode, groups)
            vals_all += vals
            pos = x - 0.4 + width * (mi + 0.5)
            if metric in BOUNDED:
                ax.scatter(pos, vals, s=64, marker=MODE_MARKERS[mode], color=MODE_COLORS[mode],
                           edgecolor=SURFACE, linewidth=1.2, zorder=3, label=T["modes"][mode])
                ax.vlines(pos, [ax.get_ylim()[0]] * len(pos), vals, color=MODE_COLORS[mode], linewidth=1.0,
                          alpha=0.35, zorder=2)
                fmt = "{:.3f}"
            else:
                ax.bar(pos, vals, width * 0.9, color=MODE_COLORS[mode], label=T["modes"][mode],
                       edgecolor=SURFACE, linewidth=1.0)
                fmt = "{:.2f}"
            for p_, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(fmt.format(v), (p_, v), xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
        ax.set_title(T["metrics"][metric], loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(g) if g != "__mean__" else T["mean"] for g in groups], fontsize=8.5)
        if metric in BOUNDED:
            _zoom_ylim(ax, vals_all, upper_cap=1.0)
            # dikey kılavuz çizgileri yeni eksen altına uzat
            lo = ax.get_ylim()[0]
            for coll in ax.collections:
                if hasattr(coll, "get_segments") and coll.get_segments():
                    segs = coll.get_segments()
                    coll.set_segments([[(sg[0][0], lo), sg[1]] for sg in segs])
        else:
            f = [v for v in vals_all if np.isfinite(v)]
            if f:
                ax.set_ylim(0, max(f) * 1.25 if max(f) > 0 else 1)
        ax.axvline(len(clips) - 0.5, color=GRID, linewidth=1.0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955), fontsize=9.5)
    fig.suptitle(T["by_clip_title"], fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "metrics_by_clip")


def plot_summary_by_mode(rows, lang, out_dir):
    """Durum bazında ortalama (klip noktalarıyla). En iyi stabilize eden durum çerçeveyle vurgulanır."""
    T = TEXT[lang]
    modes = _modes_in(rows)
    clips = sorted({r["clip_id"] for r in rows})
    metrics = PLOT_METRICS + ["FPS"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 4.8))
    for ax, metric in zip(axes, metrics):
        means, allv = [], []
        for mi, mode in enumerate(modes):
            v = [r[metric] for r in rows if r["mode"] == mode]
            m = float(np.nanmean(v)) if v else np.nan
            means.append(m)
            allv += v + [m]
            jitter_x = np.full(len(v), mi) + (np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else 0)
            if metric in BOUNDED:
                ax.scatter(jitter_x, v, s=18, color=MODE_COLORS[mode], alpha=0.45, zorder=2, linewidth=0)
                ax.scatter([mi], [m], s=90, marker=MODE_MARKERS[mode], color=MODE_COLORS[mode],
                           edgecolor=SURFACE, linewidth=1.4, zorder=3)
            else:
                ax.bar(mi, m, 0.66, color=MODE_COLORS[mode], edgecolor=SURFACE, linewidth=1.0, zorder=2)
                ax.scatter(jitter_x, v, s=20, color=SURFACE, edgecolor=INK, linewidth=0.8, zorder=3)
            if np.isfinite(m):
                ax.annotate("{:.3f}".format(m) if metric != "FPS" else "{:.0f}".format(m), (mi, m),
                            xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                            color=INK)
        cand = [(means[i], i) for i, md in enumerate(modes) if md in STABILIZING_MODES and np.isfinite(means[i])]
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([T["short"][m] for m in modes], rotation=30, ha="right", fontsize=8.5)
        ax.set_title(T["metrics"][metric], loc="left", fontsize=10)
        if metric in BOUNDED:
            _zoom_ylim(ax, allv, upper_cap=1.0)
        else:
            f = [v for v in allv if np.isfinite(v)]
            if f:
                ax.set_ylim(0, max(f) * 1.2)
        if cand and metric != "FPS":
            _, bi = (min if metric == "ResidualJitter" else max)(cand)
            lo, hi = ax.get_ylim()
            ax.axvspan(bi - 0.45, bi + 0.45, color=MODE_COLORS[modes[bi]], alpha=0.08, zorder=0, linewidth=0)
            ax.get_xticklabels()[bi].set_fontweight("bold")
            ax.get_xticklabels()[bi].set_color(INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=MODE_MARKERS[m], color="none", markerfacecolor=MODE_COLORS[m],
                      markeredgecolor=MODE_COLORS[m], markersize=9, label=T["modes"][m]) for m in modes]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93), fontsize=9)
    fig.suptitle(T["summary_title"].format(n=len(clips)), fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, out_dir, "summary_by_mode")


def plot_signals(sig_path, lang, out_dir):
    T = TEXT[lang]
    cid = os.path.splitext(os.path.basename(sig_path))[0]
    z = np.load(sig_path)
    fps = float(z["fps"])
    pred, pa = z["pred_offsets"], z["pred_alpha"]
    has_gt = "gt_jitter" in z.files
    gt = z["gt_jitter"] if has_gt else None
    n = len(pred) if not has_gt else min(len(pred), len(gt))
    t = np.arange(n) / fps
    series = [(pred[:n, 0], gt[:n, 0] if has_gt else None, T["dx"]),
              (pred[:n, 1], gt[:n, 1] if has_gt else None, T["dy"])]
    if pred.shape[1] > 2 or (has_gt and gt.shape[1] > 2):
        p_th = np.degrees(pred[:n, 2]) if pred.shape[1] > 2 else np.zeros(n)
        g_th = np.degrees(gt[:n, 2]) if has_gt and gt.shape[1] > 2 else None
        series.append((p_th, g_th, T["dtheta"]))
    series.append((pa[:n], z["gt_alpha"][:n] if has_gt else None, T["alpha"]))
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 2.5 * len(series)), sharex=True)
    for ax, (p, g, lab) in zip(axes, series):
        if g is not None:
            ax.plot(t, g, color=GT_COLOR, linewidth=1.1, label=T["gt"], alpha=0.85)
        ax.plot(t, p, color=PRED_COLOR, linewidth=1.4, label=T["pred"])
        ax.set_ylabel(lab)
        ax.grid(True, axis="both", color=GRID, linewidth=0.5)
    # kontrol deneyinin kullandığı sabit α (adaptif α'nın klip ortalaması)
    axes[-1].axhline(float(np.mean(pa)), color=MODE_COLORS["alpha_mean"], linewidth=1.3, linestyle="--",
                     label=T["modes"]["alpha_mean"] + " = " + ("{:.2f}".format(float(np.mean(pa)))
                                                               .replace(".", "," if lang == "tr" else ".")))
    axes[-1].legend(loc="lower right", fontsize=8.5)
    axes[-1].set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel(T["time"])
    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle(T["signals_title"].format(clip=cid.replace("video_", "")), fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "signals_" + cid.replace("video_", ""))


def plot_trajectories(sig_path, lang, out_dir):
    T = TEXT[lang]
    cid = os.path.splitext(os.path.basename(sig_path))[0]
    z = np.load(sig_path)
    if "gt_trajectory" not in z.files:
        return
    fps = float(z["fps"])
    pred, pa = z["pred_offsets"], z["pred_alpha"]
    traj, smooth = z["gt_trajectory"], z["gt_smooth"]
    n = min(len(pred), len(traj))
    t = np.arange(n) / fps
    dims = min(traj.shape[1], pred.shape[1])
    labels = [T["traj_x"], T["traj_y"], T["traj_th"]]
    fig, axes = plt.subplots(dims, 1, figsize=(12, 3.4 * dims), sharex=True)
    for k, ax in enumerate(axes):
        conv = np.degrees if k == 2 else (lambda v: v)
        ax.plot(t, conv(smooth[:n, k]), color=GRID, linewidth=4.0, label=T["smooth"], zorder=1)
        for mode, a in config.MODES:
            if mode == "alpha_0.0":
                continue
            if a is None:
                alpha = pa[:n]
            elif a == "mean":
                alpha = np.full(n, float(np.mean(pa)))
            else:
                alpha = np.full(n, float(a))
            corrected = traj[:n, k] - alpha * pred[:n, k]
            ax.plot(t, conv(corrected), color=MODE_COLORS[mode], linewidth=1.2, label=T["modes"][mode], zorder=2)
        ax.plot(t, conv(traj[:n, k]), color=MODE_COLORS["alpha_0.0"], linewidth=1.2,
                label=T["modes"]["alpha_0.0"] + " = " + T["orig"].lower(), zorder=3, alpha=0.9)
        ax.set_ylabel(labels[k])
        ax.grid(True, axis="both", color=GRID, linewidth=0.5)
    axes[-1].set_xlabel(T["time"])
    axes[0].legend(loc="upper left", ncol=3, fontsize=8.5)
    fig.suptitle(T["traj_title"].format(clip=cid.replace("video_", "")), fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "trajectory_" + cid.replace("video_", ""))


# ---------------------------------------------------------------------------
# Eğitim grafiği
# ---------------------------------------------------------------------------
def plot_training_history(history_csv, run_dir):
    """Eğitim/doğrulama kayıpları + en iyi epoch ve naif referans çizgileri (summary.json varsa)."""
    import json
    _style()
    rows = _read_rows(history_csv)
    ep = [r["epoch"] for r in rows]
    summ = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summ = json.load(f)
    naive = {"motion": summ.get("baseline_zero_motion_mse"), "alpha": summ.get("baseline_mean_alpha_mse")}
    best = summ.get("best_epoch")
    for lang in ("en", "tr"):
        T = TEXT[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, key, title in zip(axes, ["total", "motion", "alpha"], [T["total"], T["motion"].format(comps=_comps_label()), T["alpha_loss"]]):
            ax.plot(ep, [r["train_" + key] for r in rows], color="#2a78d6", label=T["train"])
            ax.plot(ep, [r["val_" + key] for r in rows], color="#eb6834", label=T["val"])
            if naive.get(key):
                ax.axhline(naive[key], color=INK2, linewidth=0.9, alpha=0.8)
                ax.annotate(T["naive"], (ep[-1], naive[key]), xytext=(0, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=8, color=INK2)
            if best:
                ax.axvline(best, color=INK2, linewidth=0.9, alpha=0.5)
                ax.annotate("{} ({})".format(T["best"], int(best)), (best, 0), xycoords=("data", "axes fraction"),
                            xytext=(3, 4), textcoords="offset points", fontsize=8, color=INK2)
            ax.set_title(title, loc="left")
            ax.set_xlabel(T["epoch"])
            ax.set_ylabel(T["loss_norm"])
            ax.set_yscale("log")
            ax.grid(True, axis="both", color=GRID, linewidth=0.5)
        axes[0].legend(loc="upper right")
        fig.suptitle(T["loss_title"], fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        _save(fig, os.path.join(run_dir, "plots", lang), "training_loss")


def make_all_plots(test_dir=None):
    _style()
    test_dir = test_dir or config.TEST_RESULTS_DIR
    metrics_csv = os.path.join(test_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        print("Test sonucu bulunamadı: {}  -> önce `python main.py test`".format(metrics_csv))
        return
    rows = _read_rows(metrics_csv)
    sigs = sorted(glob.glob(os.path.join(test_dir, "signals", "*.npz")))
    clip_ids = {r["clip_id"] for r in rows}
    sigs = [s for s in sigs if os.path.splitext(os.path.basename(s))[0] in clip_ids]
    for lang in ("en", "tr"):
        out = os.path.join(test_dir, "plots", lang)
        plot_metrics_by_clip(rows, lang, out)
        plot_summary_by_mode(rows, lang, out)
        for s in sigs:
            plot_signals(s, lang, out)
            plot_trajectories(s, lang, out)
    hist = os.path.join(config.TRAIN_RUN_DIR, "history.csv")
    if os.path.exists(hist):
        plot_training_history(hist, config.TRAIN_RUN_DIR)
    print("Grafikler kaydedildi: {}/plots/{{en,tr}}".format(test_dir))
    from pipeline.tables import make_all_tables
    make_all_tables(test_dir)
