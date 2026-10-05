"""
Çok alanlı eğitim: IDA_split/train + Veri Seti/Araba, Vapur, Yürüyüş (config.EXTRA_TEST_CLIPS hariç).

Akış:
 1. Train kliplerinin IMU özellikleri + RAFT ground truth'u (önbellekten ya da ilk seferde hesaplanarak) alınır.
    Alanlar arası dengesizlik (IDA ~15 dk, diğerleri ~2 dk) config.DOMAIN_BALANCE ile örneklemede dengelenir.
 2. Train klipleri KLİP BAZINDA eğitim / doğrulama olarak ayrılır (aynı klibin pencereleri iki tarafa düşmez).
 3. Her klip 60 karelik, 5 kare adımlı pencerelere bölünür. X = IMU jitter (3), Y = dx, dy, alpha.
 4. IMU girdisi eğitim istatistikleriyle standartlaştırılır (ortalama/std checkpoint'e kaydedilir).
 5. Mini-batch Adam, doğrulama kaybına göre en iyi model kaydı, LR azaltma ve erken durdurma.
"""
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import config
from models.imu_alpha_net import IMUAdaptiveStabilizerNet
from pipeline import dataset
from pipeline.device import get_device
from pipeline.model_io import save_checkpoint
from pipeline.prepare import prepare_clips


def _split_train_val(clips, data, frac, seed):
    """Klip bazında, alan bazında tabakalı doğrulama ayrımı (config.VAL_CLIPS_PER_DOMAIN).
    Çok kısa klipler (< 3 pencere) doğrulamaya gitmez."""
    rng = np.random.RandomState(seed)
    per_dom = getattr(config, "VAL_CLIPS_PER_DOMAIN", None)
    val_ids = set()
    doms = sorted({c.get("domain", "ida") for c in clips})
    for dom in doms:
        cands = [c for c in clips if c.get("domain", "ida") == dom
                 and len(data[c["clip_id"]]["imu_jitter"]) >= 2 * config.WINDOW_SIZE]
        if per_dom is None:
            k = int(round(len(cands) * frac)) if dom == "ida" else 0
        else:
            k = int(per_dom.get(dom, 0))
        k = min(k, max(0, len(cands) - 1))          # her alanda en az bir eğitim klibi kalsın
        for i in rng.permutation(len(cands))[:k]:
            val_ids.add(cands[i]["clip_id"])
    tr = [c for c in clips if c["clip_id"] not in val_ids]
    va = [c for c in clips if c["clip_id"] in val_ids]
    return tr, va


def _windows_for(clips, data):
    Xs, Ys, doms = [], [], []
    for c in clips:
        d = data[c["clip_id"]]
        X, Y = dataset.make_windows(d["imu_jitter"], d["jitter"][:, dataset.motion_index()], d["alpha"])
        if len(X):
            Xs.append(X)
            Ys.append(Y)
            doms += [c.get("domain", "ida")] * len(X)
    return np.concatenate(Xs), np.concatenate(Ys), np.array(doms)


def _domain_weights(doms, balance):
    """Pencere örnekleme ağırlıkları: alan d'nin toplam payı n_d^(1-balance) ile orantılı."""
    names, counts = np.unique(doms, return_counts=True)
    share = counts.astype(np.float64) ** (1.0 - balance)
    share /= share.sum()
    w = np.zeros(len(doms))
    for n, c, s in zip(names, counts, share):
        w[doms == n] = s / c
    return w / w.sum(), dict(zip(names, counts)), dict(zip(names, share))


def _predict(model, X, device, bs=256):
    model.eval()
    pms, pas = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pm, pa = model(X[i:i + bs].to(device))
            pms.append(pm.cpu())
            pas.append(pa.cpu())
    return torch.cat(pms), torch.cat(pas)


def _evaluate(model, X, Y, doms, device, alpha_naive):
    """Havuzlanmış doğrulama MSE'leri + alan bazında göreli hatalar ve model seçim ölçütü.
    Göreli hareket = MSE / (0 tahmininin MSE'si); göreli α = MSE / (eğitim ortalaması α tahmininin MSE'si)."""
    D = dataset.motion_dim()
    pm, pa = _predict(model, X, device)
    ym, ya = Y[:, :, :D], Y[:, :, D:D + 1]
    m = float(((pm - ym) ** 2).mean())
    a = float(((pa - ya) ** 2).mean())
    per = {}
    for dom in sorted(set(doms.tolist())):
        idx = torch.from_numpy(np.where(doms == dom)[0])
        em = float(((pm[idx] - ym[idx]) ** 2).mean())
        bm = float((ym[idx] ** 2).mean()) + 1e-12
        ea = float(((pa[idx] - ya[idx]) ** 2).mean())
        # taban: α hedefi bir alanda neredeyse sabitse naif hata ~0 olur ve oran patlar -> en az 0.01 (α std ~0.1)
        ba = max(float(((ya[idx] - alpha_naive) ** 2).mean()), 0.01)
        per[dom] = (em / bm, ea / ba)
    rel_m = float(np.mean([v[0] for v in per.values()]))
    rel_a = float(np.mean([v[1] for v in per.values()]))
    select = config.LAMBDA_MOTION * rel_m + config.LAMBDA_ALPHA * rel_a
    return {"m": m, "a": a, "total": config.LAMBDA_MOTION * m + config.LAMBDA_ALPHA * a,
            "rel_m": rel_m, "rel_a": rel_a, "select": select, "per": per}


def _train_epoch(model, opt, Xtr_t, Ytr_t, samp_w_t, batch_size, dev, D, mse, lam_alpha=None):
    lam_alpha = config.LAMBDA_ALPHA if lam_alpha is None else lam_alpha
    model.train()
    # Alan dengeli örnekleme (yerine koyarak); DOMAIN_BALANCE=0 ise düz karıştırma
    if config.DOMAIN_BALANCE > 0:
        perm = torch.multinomial(samp_w_t, len(Xtr_t), replacement=True)
    else:
        perm = torch.randperm(len(Xtr_t))
    s_m, s_a, nb = 0.0, 0.0, 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:      # BatchNorm tek örnekle çalışmaz
            continue
        xb, yb = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
        pm, pa = model(xb)
        lm = mse(pm, yb[:, :, :D])
        la = mse(pa, yb[:, :, D:D + 1])
        loss = config.LAMBDA_MOTION * lm + lam_alpha * la
        opt.zero_grad()
        loss.backward()
        opt.step()
        s_m += lm.item()
        s_a += la.item()
        nb += 1
    return s_m / nb, s_a / nb


def _oof_motion_predictions(clips, data, device, batch_size, lr):
    """alpha* hedefi için: her klibin hareketi, o klibi eğitimde görmeyen bir modelle tahmin edilir (K katlı çapraz
    doğrulama, katlar alan bazında dengeli). Dönüş: {clip_id: (N, 3) model biriminde dx, dy, dθ (tahmin edilmeyen 0)}."""
    K = max(2, int(getattr(config, "ALPHA_OPT_FOLDS", 5)))
    rng = np.random.RandomState(config.SEED + 1)
    fold = {}
    for dom in sorted({c.get("domain", "ida") for c in clips}):
        dc = [c["clip_id"] for c in clips if c.get("domain", "ida") == dom]
        off = rng.randint(K)
        for i, j in enumerate(rng.permutation(len(dc))):
            fold[dc[j]] = (i + off) % K
    idx = dataset.motion_index()
    D = len(idx)
    all_imu = np.concatenate([data[c["clip_id"]]["imu_jitter"] for c in clips])
    mean, std = all_imu.mean(0), all_imu.std(0) + 1e-8
    t_std = np.concatenate([data[c["clip_id"]]["jitter"][:, idx] for c in clips]).std(0) + 1e-8
    dev = get_device(device)
    mse = nn.MSELoss()
    preds = {}
    t0 = time.time()
    for k in range(K):
        tr = [c for c in clips if fold[c["clip_id"]] != k]
        te = [c for c in clips if fold[c["clip_id"]] == k]
        if not te:
            continue
        X, Y, doms = _windows_for(tr, data)
        X = (X - mean) / std
        Y[:, :, :D] /= t_std
        w, _, _ = _domain_weights(doms, config.DOMAIN_BALANCE)
        torch.manual_seed(config.SEED + 10 + k)
        model = IMUAdaptiveStabilizerNet(dilations=config.MODEL_DILATIONS, dropout=config.MODEL_DROPOUT,
                                         in_channels=X.shape[2], motion_dim=D).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        Xt, Yt, wt = torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(w).double()
        for _ in range(int(getattr(config, "ALPHA_OPT_EPOCHS", 30))):
            _train_epoch(model, opt, Xt, Yt, wt, batch_size, dev, D, mse, lam_alpha=0.0)
        model.eval()
        with torch.no_grad():
            for c in te:
                x = (data[c["clip_id"]]["imu_jitter"] - mean) / std
                pm, _ = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(dev))
                full = np.zeros((len(x), data[c["clip_id"]]["jitter"].shape[1]), np.float32)
                full[:, idx] = pm.squeeze(0).cpu().numpy() * t_std
                preds[c["clip_id"]] = full
    print("  alpha* için çapraz doğrulama tahminleri: {} kat, {} klip ({:.1f} sn)".format(K, len(preds), time.time() - t0))
    return preds


def train(device=None, raft_bf16=False, epochs=None, batch_size=None, lr=None):
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 70)
    print("ACCFlow | EĞİTİM (IDA_split/train + Araba/Vapur/Yürüyüş)")
    print("=" * 70)

    clips = dataset.load_clips("train")
    test_ids = set(config.TEST_CLIPS)
    assert not any(c["clip_id"] in test_ids for c in clips), "Test klibi eğitim listesinde!"
    by_dom = {}
    for c in clips:
        by_dom.setdefault(c.get("domain", "ida"), []).append(c["clip_id"])
    print("Train klibi sayısı: {}  | alanlar: {}".format(
        len(clips), {k: len(v) for k, v in sorted(by_dom.items())}))
    data = prepare_clips(clips, need_gt=True, device=device, bf16=raft_bf16)

    # Alpha hedefinin dağılımı (ALPHA_PAN_V0 ayarı için): kasıtlı hareket hızı ve pan karesi oranı
    speeds = np.concatenate([dataset.smooth_speed(data[c["clip_id"]]) for c in clips])
    alphas = np.concatenate([data[c["clip_id"]]["alpha"] for c in clips])
    if config.ALPHA_MODE == "pan":
        print("Alpha hedefi: {} | kasıtlı hareket hızı (px/kare) yüzdelikleri 50/75/90/99: {} | v0 = {}".format(
        config.ALPHA_MODE, np.round(np.percentile(speeds, [50, 75, 90, 99]), 2).tolist(), config.ALPHA_PAN_V0))
        print("  α_gt ortalaması {:.2f} | std {:.2f} | pan karesi (α_gt<0.5) %{:.1f}".format(
            alphas.mean(), alphas.std(), 100 * np.mean(alphas < 0.5)))

    if config.ALPHA_MODE == "optimal":
        # alpha* = öğrenilmiş düzeltme güveni hedefi (görülmemiş klip tahminine göre ideal düzeltme oranı)
        oof = _oof_motion_predictions(clips, data, device, batch_size, lr)
        for c in clips:
            d = data[c["clip_id"]]
            d["alpha"] = dataset.optimal_alpha(d["jitter"], oof[c["clip_id"]], dataset.half_diag(d))
        print("Alpha hedefi: optimal (alpha*) | alan bazında ortalama ± std: {}".format(
            {dom: "{:.2f} ± {:.2f}".format(np.concatenate([data[c["clip_id"]]["alpha"] for c in clips
                                                           if c.get("domain", "ida") == dom]).mean(),
                                          np.concatenate([data[c["clip_id"]]["alpha"] for c in clips
                                                          if c.get("domain", "ida") == dom]).std())
             for dom in sorted(by_dom)}))

    tr_clips, va_clips = _split_train_val(clips, data, config.VAL_FRACTION, config.SEED)
    print("Eğitim klipleri   : {}".format(len(tr_clips)))
    print("Doğrulama klipleri: {} -> {}".format(len(va_clips), [c["clip_id"] for c in va_clips]))

    Xtr, Ytr, dom_tr = _windows_for(tr_clips, data)
    Xva, Yva, dom_va = _windows_for(va_clips, data)
    samp_w, dom_n, dom_share = _domain_weights(dom_tr, config.DOMAIN_BALANCE)
    print("Alan dengesi (DOMAIN_BALANCE={}): pencere sayısı {} -> örnekleme payı {}".format(
        config.DOMAIN_BALANCE, {k: int(v) for k, v in dom_n.items()},
        {k: "%{:.0f}".format(100 * v) for k, v in dom_share.items()}))

    # IMU standardizasyonu (yalnızca eğitim kliplerinin karelerinden)
    all_imu = np.concatenate([data[c["clip_id"]]["imu_jitter"] for c in tr_clips])
    imu_mean = all_imu.mean(0).astype(np.float32)
    imu_std = (all_imu.std(0) + 1e-8).astype(np.float32)
    Xtr = (Xtr - imu_mean) / imu_std
    Xva = (Xva - imu_mean) / imu_std

    # Hedef normalizasyonu: dx, dy (px) ve dθ (rad) çok farklı ölçekte -> her kanal eğitim std'sine bölünür
    D = dataset.motion_dim()
    all_jit = np.concatenate([data[c["clip_id"]]["jitter"][:, dataset.motion_index()] for c in tr_clips])
    target_std = (all_jit.std(0) + 1e-8).astype(np.float32)
    Ytr[:, :, :D] /= target_std
    Yva[:, :, :D] /= target_std
    print("Hedef: {} | çıktı: {} + alpha | IMU girdi kanalı: {} ({}) | hedef std: {}".format(
        config.TARGET, ", ".join(dataset.motion_components()), Xtr.shape[2], config.IMU_FEATURE,
        np.round(target_std, 5).tolist()))
    print("Pencereler -> train: {}  val: {}  (pencere={}, adım={})".format(
        Xtr.shape, Xva.shape, config.WINDOW_SIZE, config.STRIDE))

    dev = get_device(device)
    print("Eğitim cihazı: {}".format(dev))
    Xtr_t, Ytr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
    Xva_t, Yva_t = torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float()
    samp_w_t = torch.from_numpy(samp_w).double()

    model = IMUAdaptiveStabilizerNet(dilations=config.MODEL_DILATIONS, dropout=config.MODEL_DROPOUT,
                                     in_channels=Xtr.shape[2], motion_dim=D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    mse = nn.MSELoss()

    # Referans (naif) taban çizgileri: her zaman 0 jitter ve eğitim ortalaması alpha tahmin etmek
    alpha_naive = float(Ytr[:, :, D].mean())
    base_motion = float((Yva[:, :, :D] ** 2).mean())
    base_alpha = float(((Yva[:, :, D] - alpha_naive) ** 2).mean())
    print("Doğrulama pencereleri alan bazında: {}".format(
        {str(k): int(v) for k, v in zip(*np.unique(dom_va, return_counts=True))}))

    def ckpt_extra(ep, ev, clips_used, refit=False):
        return {
            "epoch": ep, "val_total": ev["total"] if ev else None, "val_motion": ev["m"] if ev else None,
            "val_alpha": ev["a"] if ev else None, "val_select": ev["select"] if ev else None, "refit_all": refit,
            "train_clips": [c["clip_id"] for c in clips_used],
            "val_clips": [] if refit else [c["clip_id"] for c in va_clips],
            "window": config.WINDOW_SIZE, "flow_width": config.FLOW_WIDTH,
            "dilations": list(config.MODEL_DILATIONS), "imu_feature": config.IMU_FEATURE,
            "in_channels": int(Xtr.shape[2]), "motion_dim": int(D), "target": config.TARGET,
            "motion_components": list(dataset.motion_components()), "imu_tag": dataset.imu_tag(),
            "imu_time_offset_s": float(getattr(config, "IMU_TIME_OFFSET_S", 0.0)),
            "alpha_mode": config.ALPHA_MODE,
            "target_std": target_std.tolist(), "traj_sigma": config.TRAJ_SIGMA,
            "domains": {str(k): int(v) for k, v in dom_n.items()}, "domain_balance": float(config.DOMAIN_BALANCE),
            "unit": "short_side=FLOW_WIDTH",
        }

    os.makedirs(config.TRAIN_RUN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    history = []
    best, best_epoch, bad = float("inf"), -1, 0
    t_start = time.time()
    min_epochs = getattr(config, "MIN_EPOCHS", 0)

    for ep in range(1, epochs + 1):
        tr_m, tr_a = _train_epoch(model, opt, Xtr_t, Ytr_t, samp_w_t, batch_size, dev, D, mse)
        ev = _evaluate(model, Xva_t, Yva_t, dom_va, dev, alpha_naive)
        sched.step(ev["select"])
        cur_lr = opt.param_groups[0]["lr"]
        row = {
            "epoch": ep, "train_total": config.LAMBDA_MOTION * tr_m + config.LAMBDA_ALPHA * tr_a,
            "train_motion": tr_m, "train_alpha": tr_a,
            "val_total": ev["total"], "val_motion": ev["m"], "val_alpha": ev["a"],
            "val_select": ev["select"], "val_rel_motion": ev["rel_m"], "val_rel_alpha": ev["rel_a"], "lr": cur_lr,
        }
        for dom, (rm, ra) in ev["per"].items():
            row["rel_motion_" + dom] = rm
            row["rel_alpha_" + dom] = ra
        history.append(row)

        improved = ev["select"] < best - 1e-7
        if improved:
            best, best_epoch, bad = ev["select"], ep, 0
            save_checkpoint(config.MODEL_PATH, model, imu_mean, imu_std, extra=ckpt_extra(ep, ev, tr_clips))
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0 or improved:
            per = " ".join("{} {:.2f}/{:.2f}".format(k, v[0], v[1]) for k, v in ev["per"].items())
            print("Epoch {:4d}/{} | train M {:.4f} A {:.4f} | val seçim {:.3f} (göreli hareket/α: {}) | lr {:.1e}{}".format(
                ep, epochs, tr_m, tr_a, ev["select"], per, cur_lr, "  *" if improved else ""))
        if ep >= min_epochs and bad >= config.EARLY_STOP_PATIENCE:
            print("Erken durdurma: {} epoch boyunca doğrulama seçim ölçütü iyileşmedi.".format(bad))
            break

    elapsed = time.time() - t_start
    hist_path = os.path.join(config.TRAIN_RUN_DIR, "history.csv")
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    best_row = history[best_epoch - 1]
    print("-" * 70)
    print("En iyi epoch (doğrulama seçim ölçütü): {} | göreli hareket {:.3f}, göreli α {:.3f} (1.0 = naif)".format(
        best_epoch, best_row["val_rel_motion"], best_row["val_rel_alpha"]))

    refit_seconds = None
    if getattr(config, "REFIT_ALL", False) and va_clips:
        # Seçilen epoch sayısı ve aynı öğrenme oranı takvimiyle tüm eğitim klipleri (doğrulama dahil) üzerinde
        # sıfırdan yeniden eğitim. Normalizasyon istatistikleri ilk aşamadakiyle aynı kalır.
        print("Yeniden eğitim (REFIT_ALL): {} klip, {} epoch".format(len(tr_clips) + len(va_clips), best_epoch))
        t_refit = time.time()
        torch.manual_seed(config.SEED)
        Xall = np.concatenate([Xtr, Xva])
        Yall = np.concatenate([Ytr, Yva])
        dom_all = np.concatenate([dom_tr, dom_va])
        w_all, _, _ = _domain_weights(dom_all, config.DOMAIN_BALANCE)
        Xall_t, Yall_t = torch.from_numpy(Xall).float(), torch.from_numpy(Yall).float()
        w_all_t = torch.from_numpy(w_all).double()
        model = IMUAdaptiveStabilizerNet(dilations=config.MODEL_DILATIONS, dropout=config.MODEL_DROPOUT,
                                         in_channels=Xtr.shape[2], motion_dim=D).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        for ep in range(1, best_epoch + 1):
            for g in opt.param_groups:        # ilk aşamanın öğrenme oranı takvimi (epoch başındaki değer)
                g["lr"] = history[ep - 2]["lr"] if ep > 1 else lr
            _train_epoch(model, opt, Xall_t, Yall_t, w_all_t, batch_size, dev, D, mse)
        save_checkpoint(config.MODEL_PATH, model, imu_mean, imu_std,
                        extra=ckpt_extra(best_epoch, None, list(tr_clips) + list(va_clips), refit=True))
        refit_seconds = round(time.time() - t_refit, 1)
        print("Son model tüm eğitim klipleriyle {} epoch eğitildi ({} sn).".format(best_epoch, refit_seconds))

    summary = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
        "val_motion_mse": best_row["val_motion"],
        "val_motion_note": "motion MSE normalize edilmiş birimde (1.0 = sıfır tahmin etmekle aynı)",
        "val_alpha_mse": best_row["val_alpha"],
        "val_rel_motion": best_row["val_rel_motion"],
        "val_rel_alpha": best_row["val_rel_alpha"],
        "val_rel_per_domain": {k[len("rel_motion_"):]: [best_row[k], best_row["rel_alpha_" + k[len("rel_motion_"):]]]
                               for k in best_row if k.startswith("rel_motion_")},
        "selection": "alan ortalaması göreli hata: LAMBDA_MOTION*hareket + LAMBDA_ALPHA*α",
        "refit_all": bool(getattr(config, "REFIT_ALL", False) and va_clips),
        "refit_seconds": refit_seconds,
        "baseline_zero_motion_mse": base_motion,
        "baseline_mean_alpha_mse": base_alpha,
        "n_train_windows": int(len(Xtr)),
        "n_val_windows": int(len(Xva)),
        "train_clips": [c["clip_id"] for c in tr_clips],
        "val_clips": [c["clip_id"] for c in va_clips],
        "train_domains": {str(k): int(v) for k, v in dom_n.items()},
        "domain_balance": float(config.DOMAIN_BALANCE),
        "imu_mean": imu_mean.tolist(),
        "imu_std": imu_std.tolist(),
        "model_path": os.path.relpath(config.MODEL_PATH, config.PROJECT_ROOT),
    }
    with open(os.path.join(config.TRAIN_RUN_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    from pipeline.plots import plot_training_history
    plot_training_history(hist_path, config.TRAIN_RUN_DIR)
    from pipeline.tables import make_all_tables
    make_all_tables()

    print("-" * 70)
    print("En iyi epoch: {}  | val motion MSE {:.4f} (naif 0-tahmin: {:.4f}) | val alpha MSE {:.4f} (naif: {:.4f})".format(
        best_epoch, best_row["val_motion"], base_motion, best_row["val_alpha"], base_alpha))
    print("Alan bazında göreli hata (hareket / α; 1.0 = naif): {}".format(
        {k: [round(v[0], 3), round(v[1], 3)] for k, v in summary["val_rel_per_domain"].items()}))
    print("Model kaydedildi: {}".format(config.MODEL_PATH))
    print("Eğitim çıktıları: {}".format(config.TRAIN_RUN_DIR))
    return summary
