"""
Tezde kullanılabilecek tablolar (TR + EN; Markdown, LaTeX/booktabs ve CSV).

results/multidomain/tables/{tr,en}/
  summary.*            : durum bazında ortalama ± std (tüm test klipleri) — ana sonuç tablosu
  summary_ida.*        : yalnızca IDA test klipleri (hafif sarsıntı)
  summary_shaky.*      : yalnızca araba + vapur test klipleri (belirgin sarsıntı)
  paired.*             : adaptif α vs sabit α=1 ve vs sabit α=ort(adaptif) — kazanç/eşit/kayıp + işaret testi
  per_clip_<metrik>.*  : her metrik için klip x durum tablosu
  adaptive_alpha.*     : klip bazında adaptif α istatistikleri ve hedef α* ile ilişkisi
  accuracy.*           : IMU tahmininin RAFT hedefine göre doğruluğu (klip bazında)
  training.*           : eğitim özeti (veri, ayarlar, seçim, doğrulama hataları)
  all_tables.tex / all_tables.md : hepsi tek dosyada (LaTeX: \\input{all_tables} ile eklenebilir)

`run_test.py`, `run_tables.py` ve `python main.py plot` bu tabloları üretir (RAFT / eğitim gerekmez).
LaTeX: \\usepackage{booktabs} ve geniş tablolar için \\usepackage{graphicx} (\\resizebox) gerekir.
"""
import csv
import json
import math
import os

import numpy as np

import config

OUT_DIR = os.path.join(os.path.dirname(config.TEST_RESULTS_DIR), "tables")   # results/multidomain/tables

MODEL_MODES = [m for m, _ in config.MODES]
REF_MODES = [m for m, _ in getattr(config, "REFERENCE_MODES", [])]

# Tablolarda klip / alan sırası
DOMAIN_ORDER = ["ida", "araba", "vapur", "walk"]
SHAKY_DOMAINS = ("araba", "vapur", "walk")

T = {
    "en": {
        "mode": "Mode", "clip": "Clip", "metric": "Metric", "value": "Value", "mean": "Mean",
        "domains": {"ida": "IDA", "araba": "Car", "vapur": "Ferry", "walk": "Walk"},
        "modes": {"alpha_0.0": "α = 0 (no stabilization)", "alpha_0.5": "α = 0.5", "alpha_1.0": "α = 1",
                  "alpha_mean": "α = mean of adaptive (control)", "adaptive": "Adaptive α (proposed)",
                  "raft_gt": "RAFT ref. (α = 1)", "raft_gt_alpha": "RAFT ref. (α = GT)",
                  "raft_gt_obs": "RAFT ref. (dy, dθ only; α = 1)", "imu_alpha_opt": "IMU + ideal α* (upper bound)"},
        "short": {"alpha_0.0": "α = 0", "alpha_0.5": "α = 0.5", "alpha_1.0": "α = 1", "alpha_mean": "α = mean",
                  "adaptive": "Adaptive α", "raft_gt": "RAFT", "raft_gt_alpha": "RAFT α=GT",
                  "raft_gt_obs": "RAFT dy,dθ", "imu_alpha_opt": "IMU + α*"},
        "short_note": "Columns: α = mean = control (fixed α equal to the clip mean of adaptive α); RAFT = RAFT reference "
                      "with α = 1; RAFT dy,dθ = RAFT reference correcting only dy and dθ; IMU + α* = IMU prediction with "
                      "the ideal α* (upper bound).",
        "metrics": {"Stability": "Stability ↑", "CropArea": "Crop area ↑", "Distortion": "Distortion ↑",
                    "ResidualJitter": "Residual jitter (px) ↓", "FPS": "FPS ↑"},
        "summary_cap": "Mean ± std over {n} unseen test clips ({groups}). Stabilization uses the IMU only; rows below "
                       "the line are references computed with the RAFT training target (not available at test time). "
                       "Residual jitter of RAFT ref. (α = 1) is 0 by definition. Best stabilizing mode in bold.",
        "summary_ida_cap": "Mean ± std over the {n} unseen IDA test clips (mild shake). Same conventions as the main table.",
        "summary_shaky_cap": "Mean ± std over the {n} unseen strongly shaking test clips ({groups}). Same conventions "
                             "as the main table; with n = {n} the std is only indicative.",
        "per_clip_cap": "{metric} per unseen test clip. Rows are grouped by domain; references use the RAFT target.",
        "cmp": "Comparison", "vs": "Adaptive α vs {}", "control": "control",
        "paired_cap": "Per-clip paired comparison over {n} unseen test clips: number of clips where adaptive α is "
                      "better / equal / worse and one-sided sign test p (ties excluded, |Δ| < 10⁻⁴ counted as equal). "
                      "The control (fixed α equal to the clip mean of adaptive α) applies the same average correction "
                      "without time variation.",
        "alpha_cols": ["Clip", "mean α", "std α (time)", "min α", "max α", "mean α*", "corr(α, α*)"],
        "alpha_cap": "Adaptive α predicted from the IMU per test clip. Its mean is the fixed α of the control mode. "
                     "α* is the ideal correction ratio for this model's prediction (computed from RAFT, evaluation only); "
                     "corr(α, α*) measures how well α follows it over time.",
        "acc_cap": "Accuracy of the IMU-only prediction against the RAFT target on unseen test clips "
                   "(RMSE in px at short side = {w} px; dθ in degrees). α is compared with its target α*. "
                   "dx is not predicted (yaw is not observable with an accelerometer only).",
        "acc_cols": {"Corr_dx": "corr dx", "Corr_dy": "corr dy", "Corr_dtheta": "corr dθ",
                     "RMSE_dx_px": "RMSE dx (px)", "RMSE_dy_px": "RMSE dy (px)", "RMSE_dtheta_deg": "RMSE dθ (°)",
                     "MAE_alpha": "α MAE", "Corr_alpha": "corr(α, α*)"},
        "train_cap": "Training setup and result. Relative errors are validation MSE divided by the naive baseline "
                     "(motion: predict 0; α: predict the training mean); 1.0 = naive.",
        "train_rows": {
            "clips": "Training clips (IDA / car / ferry / walk)", "val_clips": "  of which validation",
            "windows": "Windows (train / validation)", "inputs": "Model input", "outputs": "Model output",
            "alpha_target": "α target", "imu_offset": "IMU time offset", "balance": "Domain balance (sampling share)",
            "selection": "Model selection", "best_epoch": "Selected epoch / epochs run",
            "refit": "Final model", "val_motion": "Validation motion MSE (normalized) / naive",
            "val_alpha": "Validation α MSE / naive", "rel": "Relative error {dom} (motion / α)",
            "train_seconds": "Training time (s)"},
        "alpha_target_opt": "α* (ideal correction ratio, out-of-fold)",
        "alpha_target_pan": "pan detection (v0 = {v0} px/frame)",
        "selection_txt": "domain-averaged relative error",
        "refit_txt": "retrained on all {n} clips",
        "refit_no": "best validation checkpoint",
    },
    "tr": {
        "mode": "Durum", "clip": "Klip", "metric": "Metrik", "value": "Değer", "mean": "Ortalama",
        "domains": {"ida": "IDA", "araba": "Araba", "vapur": "Vapur", "walk": "Yürüyüş"},
        "modes": {"alpha_0.0": "α = 0 (stabilizasyon yok)", "alpha_0.5": "α = 0,5", "alpha_1.0": "α = 1",
                  "alpha_mean": "α = adaptif ortalaması (kontrol)", "adaptive": "Adaptif α (önerilen)",
                  "raft_gt": "RAFT ref. (α = 1)", "raft_gt_alpha": "RAFT ref. (α = GT)",
                  "raft_gt_obs": "RAFT ref. (yalnız dy, dθ; α = 1)", "imu_alpha_opt": "IMU + ideal α* (üst sınır)"},
        "short": {"alpha_0.0": "α = 0", "alpha_0.5": "α = 0,5", "alpha_1.0": "α = 1", "alpha_mean": "α = ort.",
                  "adaptive": "Adaptif α", "raft_gt": "RAFT", "raft_gt_alpha": "RAFT α=GT",
                  "raft_gt_obs": "RAFT dy,dθ", "imu_alpha_opt": "IMU + α*"},
        "short_note": "Sütunlar: α = ort. = kontrol (adaptif α'nın klip ortalamasına eşit sabit α); RAFT = α = 1 ile "
                      "RAFT referansı; RAFT dy,dθ = yalnızca dy ve dθ'yı düzelten RAFT referansı; IMU + α* = IMU "
                      "tahmini ve ideal α* (üst sınır).",
        "metrics": {"Stability": "Kararlılık ↑", "CropArea": "Kalan alan ↑", "Distortion": "Bozulma ↑",
                    "ResidualJitter": "Kalan sarsıntı (px) ↓", "FPS": "FPS ↑"},
        "summary_cap": "{n} görülmemiş test klibinin ({groups}) ortalaması ± standart sapması. Stabilizasyon yalnızca "
                       "IMU ile yapılır; çizginin altındaki satırlar RAFT eğitim hedefiyle hesaplanan referanslardır "
                       "(testte kullanılamaz). RAFT ref. (α = 1) için kalan sarsıntı tanım gereği 0'dır. "
                       "En iyi stabilize eden durum kalın.",
        "summary_ida_cap": "Yalnızca {n} görülmemiş IDA test klibinin (hafif sarsıntı) ortalaması ± standart sapması. "
                           "Gösterim ana tabloyla aynıdır.",
        "summary_shaky_cap": "Belirgin sarsıntılı {n} görülmemiş test klibinin ({groups}) ortalaması ± standart sapması. "
                             "Gösterim ana tabloyla aynıdır; n = {n} olduğundan standart sapma yalnızca fikir verir.",
        "per_clip_cap": "Görülmemiş test kliplerinde {metric}. Satırlar alana göre gruplanmıştır; referanslar RAFT "
                        "hedefini kullanır.",
        "cmp": "Karşılaştırma", "vs": "Adaptif α – {}", "control": "kontrol",
        "paired_cap": "{n} görülmemiş test klibinde eşleştirilmiş karşılaştırma: adaptif α'nın daha iyi / eşit / "
                      "daha kötü olduğu klip sayısı ve tek yönlü işaret testi p değeri (eşitlikler hariç; |Δ| < 10⁻⁴ "
                      "eşit sayılır). Kontrol durumu (adaptif α'nın klip ortalamasına eşit sabit α) aynı ortalama "
                      "düzeltmeyi zamanda değiştirmeden uygular.",
        "alpha_cols": ["Klip", "ortalama α", "α std (zamanda)", "min α", "maks α", "ortalama α*", "kor(α, α*)"],
        "alpha_cap": "Test kliplerinde IMU'dan tahmin edilen adaptif α. Ortalaması kontrol durumunun sabit α'sıdır. "
                     "α*, bu modelin tahmini için ideal düzeltme oranıdır (RAFT'tan, yalnızca değerlendirme); "
                     "kor(α, α*) α'nın onu zamanda ne kadar izlediğini gösterir.",
        "acc_cap": "Yalnızca IMU ile yapılan tahminin görülmemiş test kliplerinde RAFT hedefine göre doğruluğu "
                   "(RMSE: kısa kenar = {w} px ölçeğinde px; dθ derece). α, hedefi α* ile karşılaştırılır. "
                   "dx tahmin edilmez (yalnızca ivmeölçerle yaw gözlenemez).",
        "acc_cols": {"Corr_dx": "kor. dx", "Corr_dy": "kor. dy", "Corr_dtheta": "kor. dθ",
                     "RMSE_dx_px": "RMSE dx (px)", "RMSE_dy_px": "RMSE dy (px)", "RMSE_dtheta_deg": "RMSE dθ (°)",
                     "MAE_alpha": "α MAE", "Corr_alpha": "kor(α, α*)"},
        "train_cap": "Eğitim düzeni ve sonucu. Göreli hata = doğrulama MSE / naif referans (hareket: 0 tahmini; "
                     "α: eğitim ortalaması); 1,0 = naif.",
        "train_rows": {
            "clips": "Eğitim klibi (IDA / araba / vapur / yürüyüş)", "val_clips": "  bunlardan doğrulama",
            "windows": "Pencere (eğitim / doğrulama)", "inputs": "Model girdisi", "outputs": "Model çıktısı",
            "alpha_target": "α hedefi", "imu_offset": "IMU zaman kayması", "balance": "Alan dengesi (örnekleme payı)",
            "selection": "Model seçimi", "best_epoch": "Seçilen epoch / çalışan epoch",
            "refit": "Son model", "val_motion": "Doğrulama hareket MSE (normalize) / naif",
            "val_alpha": "Doğrulama α MSE / naif", "rel": "Göreli hata {dom} (hareket / α)",
            "train_seconds": "Eğitim süresi (sn)"},
        "alpha_target_opt": "α* (ideal düzeltme oranı, çapraz doğrulamalı)",
        "alpha_target_pan": "pan algılama (v0 = {v0} px/kare)",
        "selection_txt": "alan ortalaması göreli hata",
        "refit_txt": "tüm {n} klip ile yeniden eğitildi",
        "refit_no": "en iyi doğrulama noktası",
    },
}

# "En iyi stabilize eden durum" adayları (kalın yazılır; α = 0 ve referanslar hariç)
STABILIZING_MODES = ("alpha_0.5", "alpha_1.0", "alpha_mean", "adaptive")
PAIRED_REFS = ("alpha_1.0", "alpha_mean")
TIE_TOL = 1e-4

DECIMALS = {"Stability": 3, "CropArea": 3, "Distortion": 3, "ResidualJitter": 2, "FPS": 0}
HIGHER_BETTER = {"Stability": True, "CropArea": True, "Distortion": True, "ResidualJitter": False, "FPS": True}


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _domain(cid):
    for p in ("araba", "vapur", "walk"):
        if str(cid).startswith(p + "_"):
            return p
    return "ida"


def _clip_order(cids):
    return sorted(cids, key=lambda c: (DOMAIN_ORDER.index(_domain(c)) if _domain(c) in DOMAIN_ORDER else 99, c))


def _clip_label(cid, lang):
    dom = _domain(cid)
    s = str(cid)
    s = s[len("video_"):] if s.startswith("video_") else s[len(dom) + 1:]
    s = s.replace("_part", " p")
    return "{} {}".format(T[lang]["domains"][dom], s)


def _num(v, d, lang):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    s = "{:.{d}f}".format(v, d=d)
    return s.replace(".", ",") if lang == "tr" else s


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _groups_text(cids, lang):
    L = T[lang]
    cnt = {}
    for c in cids:
        cnt[_domain(c)] = cnt.get(_domain(c), 0) + 1
    return ", ".join("{} {}".format(cnt[d], L["domains"][d]) for d in DOMAIN_ORDER if d in cnt)


def _esc(t):
    return (t.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_")
            .replace("±", "$\\pm$").replace("α*", "$\\alpha^*$").replace("α", "$\\alpha$")
            .replace("θ", "$\\theta$").replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
            .replace("°", "$^\\circ$").replace("⁻⁴", "$^{-4}$").replace("Δ", "$\\Delta$"))


def _write(out_dir, name, header, body, caption, bold=None, rules=None, collect=None, align=None):
    """body: satır listesi (hücre metinleri). bold: {(satır, sütun)} kalın hücreler.
    rules: bu satır indekslerinden ÖNCE \\midrule (yalnızca LaTeX). collect: all_tables için liste."""
    bold, rules = bold or set(), set(rules or [])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    for i, row in enumerate(body):
        cells = ["**{}**".format(c) if (i, j) in bold and c not in ("", "–") else c for j, c in enumerate(row)]
        md.append("| " + " | ".join(cells) + " |")
    md_text = "\n".join(md) + "\n\n" + caption + "\n"
    with open(os.path.join(out_dir, name + ".md"), "w") as f:
        f.write(md_text)
    # Tablo sayfadan genişse küçültülür, dar tablolar olduğu gibi kalır (graphicx)
    tex = ["\\begin{table}[ht]", "\\centering", "\\small",
           "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%"]
    tex += ["\\begin{tabular}{" + (align or "l" + "r" * (len(header) - 1)) + "}", "\\toprule",
            " & ".join(_esc(h) for h in header) + " \\\\", "\\midrule"]
    for i, row in enumerate(body):
        if i in rules:
            tex.append("\\midrule")
        cells = ["\\textbf{" + _esc(c) + "}" if (i, j) in bold and c not in ("", "–") else _esc(c)
                 for j, c in enumerate(row)]
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "}"]
    tex += ["\\caption{" + _esc(caption) + "}", "\\label{tab:" + name + "}", "\\end{table}"]
    tex_text = "\n".join(tex) + "\n"
    with open(os.path.join(out_dir, name + ".tex"), "w") as f:
        f.write(tex_text)
    if collect is not None:
        collect.append((name, md_text, tex_text))


# ---------------------------------------------------------------------------
# Tablolar
# ---------------------------------------------------------------------------
def summary_table(rows, lang, out_dir, name="summary", cap_key="summary_cap", collect=None):
    L = T[lang]
    if not rows:
        return
    metrics = [m for m in ["Stability", "CropArea", "Distortion", "ResidualJitter", "FPS"] if m in rows[0]]
    present = {r["mode"] for r in rows}
    modes = [m for m in MODEL_MODES + REF_MODES if m in present]
    cids = {r["clip_id"] for r in rows}
    n = len(cids)
    stats = {}
    for m in modes:
        for k in metrics:
            v = np.array([r[k] for r in rows if r["mode"] == m and isinstance(r.get(k), float)], float)
            v = v[np.isfinite(v)]
            stats[m, k] = (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
    bold = set()
    for j, k in enumerate(metrics):
        cand = [(stats[m, k][0], i) for i, m in enumerate(modes)
                if m in STABILIZING_MODES and np.isfinite(stats[m, k][0])]
        if cand and k != "FPS":
            _, i = (max if HIGHER_BETTER[k] else min)(cand)
            bold.add((i, j + 1))
    body = []
    for m in modes:
        nm = len({r["clip_id"] for r in rows if r["mode"] == m})
        row = [L["modes"][m] + ("" if nm == n else " (n={})".format(nm))]
        for k in metrics:
            mu, sd = stats[m, k]
            row.append("–" if not np.isfinite(mu) else "{} ± {}".format(
                _num(mu, DECIMALS[k], lang), _num(sd, DECIMALS[k], lang)))
        body.append(row)
    first_ref = [i for i, m in enumerate(modes) if m in REF_MODES]
    _write(out_dir, name, [L["mode"]] + [L["metrics"][k] for k in metrics], body,
           L[cap_key].format(n=n, groups=_groups_text(cids, lang)), bold,
           rules=first_ref[:1], collect=collect)


def per_clip_table(rows, lang, out_dir, collect=None):
    L = T[lang]
    present = {r["mode"] for r in rows}
    modes = [m for m in MODEL_MODES + REF_MODES if m in present]
    clips = _clip_order({r["clip_id"] for r in rows})
    by = {(r["clip_id"], r["mode"]): r for r in rows}
    for k in ["Stability", "CropArea", "Distortion", "ResidualJitter"]:
        if k not in rows[0]:
            continue
        body, rules, prev = [], [], None
        for c in clips:
            if prev is not None and _domain(c) != prev:
                rules.append(len(body))
            prev = _domain(c)
            row = [_clip_label(c, lang)]
            for m in modes:
                v = by.get((c, m), {}).get(k)
                row.append(_num(v, DECIMALS[k], lang) if isinstance(v, float) else "–")
            body.append(row)
        rules.append(len(body))
        mean_row = [L["mean"]]
        for m in modes:
            v = np.array([r[k] for r in rows if r["mode"] == m and isinstance(r.get(k), float)], float)
            mean_row.append(_num(float(np.nanmean(v)), DECIMALS[k], lang) if len(v) and np.isfinite(v).any() else "–")
        body.append(mean_row)
        _write(out_dir, "per_clip_" + k.lower(), [L["clip"]] + [L["short"][m] for m in modes], body,
               L["per_clip_cap"].format(metric=L["metrics"][k]) + " " + L["short_note"], rules=rules, collect=collect)


def _sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2.0 ** n


def paired_table(rows, lang, out_dir, collect=None):
    L = T[lang]
    by = {(r["clip_id"], r["mode"]): r for r in rows}
    clips = sorted({r["clip_id"] for r in rows})
    metrics = [k for k in ["Stability", "CropArea", "Distortion", "ResidualJitter"] if k in rows[0]]
    body, bold = [], set()
    for ref in PAIRED_REFS:
        pairs = [(by[(c, "adaptive")], by[(c, ref)]) for c in clips if (c, "adaptive") in by and (c, ref) in by]
        if not pairs:
            continue
        row = [L["vs"].format(L["control"] if ref == "alpha_mean" else L["short"][ref])]
        for j, k in enumerate(metrics):
            w = t = l = 0
            for ra, rb in pairs:
                a, b = ra.get(k), rb.get(k)
                if not isinstance(a, float) or not isinstance(b, float) or not (np.isfinite(a) and np.isfinite(b)):
                    continue
                diff = (a - b) if HIGHER_BETTER[k] else (b - a)
                if abs(diff) < TIE_TOL:
                    t += 1
                elif diff > 0:
                    w += 1
                else:
                    l += 1
            p = _sign_test_p(w, l)
            if np.isfinite(p) and p < 0.05:
                bold.add((len(body), j + 1))
            row.append("{}/{}/{} (p={})".format(w, t, l, _num(p, 3, lang)) if w + t + l else "–")
        body.append(row)
    if body:
        _write(out_dir, "paired", [L["cmp"]] + [L["metrics"][k] for k in metrics], body,
               L["paired_cap"].format(n=len(clips)) + (" p < 0,05 kalın." if lang == "tr" else " p < 0.05 in bold."),
               bold, collect=collect)


def adaptive_alpha_table(lang, out_dir, test_dir=None, collect=None):
    L = T[lang]
    sig_dir = os.path.join(test_dir or config.TEST_RESULTS_DIR, "signals")
    if not os.path.isdir(sig_dir):
        return
    files = {fn[:-4]: os.path.join(sig_dir, fn) for fn in os.listdir(sig_dir) if fn.endswith(".npz")}
    body, stats, rules, prev = [], [], [], None
    for cid in _clip_order(files):
        z = np.load(files[cid])
        pa = z["pred_alpha"].astype(float)
        st = [pa.mean(), pa.std(), pa.min(), pa.max()]
        if "gt_alpha" in z.files:
            ga = z["gt_alpha"].astype(float)
            n = min(len(pa), len(ga))
            c = float(np.corrcoef(pa[:n], ga[:n])[0, 1]) if np.std(pa[:n]) > 1e-9 and np.std(ga[:n]) > 1e-9 else np.nan
            st += [ga[:n].mean(), c]
        else:
            st += [np.nan, np.nan]
        if prev is not None and _domain(cid) != prev:
            rules.append(len(body))
        prev = _domain(cid)
        stats.append(st)
        body.append([_clip_label(cid, lang)] + [_num(v, 2 if i == 5 else 3, lang) for i, v in enumerate(st)])
    if body:
        rules.append(len(body))
        m = np.nanmean(np.array(stats, float), axis=0)
        body.append([L["mean"]] + [_num(v, 2 if i == 5 else 3, lang) for i, v in enumerate(m)])
        _write(out_dir, "adaptive_alpha", L["alpha_cols"], body, L["alpha_cap"], rules=rules, collect=collect)


def accuracy_table(acc_rows, lang, out_dir, collect=None):
    L = T[lang]
    if not acc_rows:
        return
    cand = [("Corr_dx", 2), ("Corr_dy", 2), ("Corr_dtheta", 2), ("RMSE_dx_px", 3), ("RMSE_dy_px", 3),
            ("RMSE_dtheta_deg", 3), ("MAE_alpha", 3), ("Corr_alpha", 2)]
    # dx tahmin edilmiyorsa (korelasyon hep NaN) dx sütunları çıkarılır: RMSE dx yalnızca hedefin kendisi olurdu
    dx_pred = any(isinstance(a.get("Corr_dx"), float) and np.isfinite(a["Corr_dx"]) for a in acc_rows)
    keys = [(k, d) for k, d in cand if (dx_pred or not k.endswith("_dx") and k != "RMSE_dx_px")
            and any(isinstance(a.get(k), float) for a in acc_rows)]
    body, rules, prev = [], [], None
    rows = {str(a["clip_id"]): a for a in acc_rows}
    for cid in _clip_order(rows):
        a = rows[cid]
        if prev is not None and _domain(cid) != prev:
            rules.append(len(body))
        prev = _domain(cid)
        body.append([_clip_label(cid, lang)] + [_num(a.get(k), d, lang) if isinstance(a.get(k), float) else "–"
                                               for k, d in keys])
    rules.append(len(body))
    mean_row = [L["mean"]]
    for k, d in keys:
        v = np.array([a[k] for a in acc_rows if isinstance(a.get(k), float)], float)
        mean_row.append(_num(float(np.nanmean(v)), d, lang) if len(v) and np.isfinite(v).any() else "–")
    body.append(mean_row)
    _write(out_dir, "accuracy", [L["clip"]] + [L["acc_cols"][k] for k, _ in keys], body,
           L["acc_cap"].format(w=config.FLOW_WIDTH), rules=rules, collect=collect)


def training_table(summ, lang, out_dir, collect=None):
    L = T[lang]
    R = L["train_rows"]
    from pipeline import dataset
    tr = list(summ.get("train_clips", []))
    va = list(summ.get("val_clips", []))
    allc = tr + va

    def cnt(cs):
        return " / ".join(str(sum(1 for c in cs if _domain(c) == d)) for d in DOMAIN_ORDER)

    comps = summ.get("motion_components") or list(dataset.motion_components())
    outputs = ", ".join({"dx": "dx", "dy": "dy", "dtheta": "dθ"}[c] for c in comps) + ", α"
    inputs = "IMU ({}, {} {})".format(config.IMU_FEATURE, len(summ.get("imu_mean", [])),
                                      "kanal" if lang == "tr" else "channels")
    if config.ALPHA_MODE == "optimal":
        a_target = L["alpha_target_opt"]
    else:
        a_target = L["alpha_target_pan"].format(v0=_num(config.ALPHA_PAN_V0, 2, lang))
    doms = summ.get("train_domains", {})
    share = np.array([float(doms.get(d, 0)) for d in DOMAIN_ORDER]) ** (1.0 - float(summ.get("domain_balance", 0)))
    share = share / share.sum() if share.sum() else share
    balance = "{} ({})".format(_num(float(summ.get("domain_balance", 0)), 1, lang),
                               " / ".join("%" + str(int(round(100 * s))) if lang == "tr" else str(int(round(100 * s))) + "%"
                                          for s in share))
    vm, nm = summ.get("val_motion_mse"), summ.get("baseline_zero_motion_mse")
    vaa, na = summ.get("val_alpha_mse"), summ.get("baseline_mean_alpha_mse")
    body = [
        [R["clips"], cnt(allc)],
        [R["val_clips"], cnt(va)],
        [R["windows"], "{} / {}".format(summ.get("n_train_windows", "–"), summ.get("n_val_windows", "–"))],
        [R["inputs"], inputs],
        [R["outputs"], outputs],
        [R["alpha_target"], a_target],
        [R["imu_offset"], "{} {}".format(_num(float(getattr(config, "IMU_TIME_OFFSET_S", 0.0)), 2, lang),
                                         "sn" if lang == "tr" else "s")],
        [R["balance"], balance],
        [R["selection"], L["selection_txt"]],
        [R["best_epoch"], "{} / {}".format(summ.get("best_epoch", "–"), summ.get("epochs_run", "–"))],
        [R["refit"], L["refit_txt"].format(n=len(allc)) if summ.get("refit_all") else L["refit_no"]],
        [R["val_motion"], "{} / {}".format(_num(vm, 3, lang), _num(nm, 3, lang)) if vm is not None else "–"],
        [R["val_alpha"], "{} / {}".format(_num(vaa, 4, lang), _num(na, 4, lang)) if vaa is not None else "–"],
    ]
    for d, (rm, ra) in sorted((summ.get("val_rel_per_domain") or {}).items(),
                              key=lambda kv: DOMAIN_ORDER.index(kv[0]) if kv[0] in DOMAIN_ORDER else 99):
        body.append([R["rel"].format(dom=L["domains"].get(d, d)), "{} / {}".format(_num(rm, 2, lang), _num(ra, 2, lang))])
    body.append([R["train_seconds"], _num(summ.get("train_seconds"), 1, lang) if summ.get("train_seconds") else "–"])
    _write(out_dir, "training", [L["metric"], L["value"]], body, L["train_cap"], collect=collect, align="ll")


def _write_all(out, collect, lang):
    order = ["summary", "summary_ida", "summary_shaky", "paired", "adaptive_alpha", "accuracy", "training",
             "per_clip_stability", "per_clip_croparea", "per_clip_distortion", "per_clip_residualjitter"]
    items = sorted(collect, key=lambda t: order.index(t[0]) if t[0] in order else 99)
    head = ("% ACCFlow tez tabloları — gerekli paketler: booktabs, graphicx\n" if lang == "tr"
            else "% ACCFlow thesis tables — requires booktabs, graphicx\n")
    with open(os.path.join(out, "all_tables.tex"), "w") as f:
        f.write(head + "\n".join("% --- {} ---\n{}".format(n, t) for n, _, t in items))
    with open(os.path.join(out, "all_tables.md"), "w") as f:
        f.write("\n".join("## {}\n\n{}".format(n, m) for n, m, _ in items))


def make_all_tables(test_dir=None):
    test_dir = test_dir or config.TEST_RESULTS_DIR
    same = os.path.abspath(test_dir) == os.path.abspath(config.TEST_RESULTS_DIR)
    out_root = OUT_DIR if same else os.path.join(os.path.dirname(os.path.abspath(test_dir)), "tables")
    made = False
    for lang in ("tr", "en"):
        out = os.path.join(out_root, lang)
        collect = []
        mpath = os.path.join(test_dir, "metrics.csv")
        if os.path.exists(mpath):
            rows = _read_csv(mpath)
            summary_table(rows, lang, out, collect=collect)
            ida = [r for r in rows if _domain(r["clip_id"]) == "ida"]
            shaky = [r for r in rows if _domain(r["clip_id"]) in SHAKY_DOMAINS]
            if ida and shaky:
                summary_table(ida, lang, out, "summary_ida", "summary_ida_cap", collect)
                summary_table(shaky, lang, out, "summary_shaky", "summary_shaky_cap", collect)
            per_clip_table(rows, lang, out, collect)
            paired_table(rows, lang, out, collect)
            adaptive_alpha_table(lang, out, test_dir, collect)
            made = True
        apath = os.path.join(test_dir, "prediction_accuracy.csv")
        if os.path.exists(apath):
            accuracy_table(_read_csv(apath), lang, out, collect)
        spath = os.path.join(config.TRAIN_RUN_DIR, "summary.json")
        if os.path.exists(spath):
            with open(spath) as f:
                training_table(json.load(f), lang, out, collect)
            made = True
        if collect:
            _write_all(out, collect, lang)
    if made:
        print("Tablolar kaydedildi: {}/{{tr,en}}  (.md / .tex / .csv + all_tables)".format(out_root))
