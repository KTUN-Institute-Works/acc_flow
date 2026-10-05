"""
IDA_split veri setini okur ve her klip için (video + IMU CSV) şu özellikleri üretir / önbelleğe alır:

  imu_jitter   (N, 3)  : stage1 (senkronizasyon) + stage2 (Butterworth high-pass) -> jitter_x/y/z
  shifts       (N-1,2) : RAFT medyan global kayma (stage3)
  trajectory   (N, 2)  : kümülatif yörünge, kare 0 = (0,0)
  smooth       (N, 2)  : Gaussian yumuşatılmış yörünge (stage4)
  jitter       (N, 2)  : görsel sarsıntı = trajectory - smooth   -> hedef dx, dy
  alpha        (N,)    : |jitter| / (|jitter| + |smooth hareket| + eps) -> hedef alpha

Önbellek: data/cache/<clip_id>.npz  (RAFT her klip için yalnızca bir kez çalışır)
"""
import os

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

import config

CACHE_VERSION = 3   # v3: RAFT akışına benzerlik (öteleme + dönme), köşe noktalarında örneklenerek


# ---------------------------------------------------------------------------
# Manifest / klip listesi
# ---------------------------------------------------------------------------
def load_clips(split=None):
    """manifest.csv (IDA_split) + ek alanlar (config.EXTRA_DOMAINS). Her eleman dict: clip_id, split, domain,
    video, sensor, start_s."""
    if not os.path.exists(config.MANIFEST_PATH):
        raise FileNotFoundError(
            "manifest.csv bulunamadı: {}\nIDA_split klasörünü data/IDA_split altına kopyalayın.".format(
                config.MANIFEST_PATH))
    df = pd.read_csv(config.MANIFEST_PATH)
    clips = []
    for _, r in df.iterrows():
        if split is not None and r["split"] != split:
            continue
        video = os.path.join(config.DATASET_DIR, r["video_file"])
        clips.append({
            "clip_id": os.path.splitext(os.path.basename(video))[0],
            "original_id": str(r["original_id"]),
            "split": r["split"],
            "video": video,
            "sensor": os.path.join(config.DATASET_DIR, r["sensor_file"]),
            "start_s": float(r["start_s"]),
            "domain": "ida",
        })
    # Ek alanlar (araba, vapur, yürüyüş)
    clips += extra_clips(split)
    return clips


def extra_clips(split=None):
    """Veri Seti/<Araba|Vapur|Yürüyüş> kayıtları (config.EXTRA_DOMAINS). Klip adı <önek>_<id>;
    config.EXTRA_TEST_CLIPS içindekiler "test", diğerleri "train"."""
    out = []
    for prefix, folder in getattr(config, "EXTRA_DOMAINS", {}).items():
        d = os.path.join(config.VERI_SETI_DIR, folder)
        if not os.path.isdir(d):
            print("UYARI: ek veri klasörü bulunamadı, atlanıyor: {}".format(d))
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("video_") and fn.endswith(".mp4")):
                continue
            rid = fn[len("video_"):-len(".mp4")]
            sensor = os.path.join(d, "sensors_{}.csv".format(rid))
            if not os.path.exists(sensor):
                print("UYARI: sensör dosyası yok, atlanıyor: {}".format(sensor))
                continue
            cid = "{}_{}".format(prefix, rid)
            sp = "test" if cid in config.EXTRA_TEST_CLIPS else "train"
            if split is not None and sp != split:
                continue
            out.append({"clip_id": cid, "original_id": cid, "split": sp, "domain": prefix,
                        "video": os.path.join(d, fn), "sensor": sensor, "start_s": None})
    return out


def domain_of(clip_id):
    """Klip adından alan: araba_/vapur_/walk_ önekleri, diğerleri IDA."""
    for prefix in getattr(config, "EXTRA_DOMAINS", {}):
        if str(clip_id).startswith(prefix + "_"):
            return prefix
    return "ida"


def select_test_clips(clip_ids=None, n=config.NUM_TEST_CLIPS):
    """Test kliplerini seçer (IDA_split/test + config.EXTRA_TEST_CLIPS). Verilmezse config.TEST_CLIPS,
    o da boşsa farklı kayıtlardan ilk n klip."""
    test = load_clips("test")
    by_id = {c["clip_id"]: c for c in test}
    wanted = clip_ids if clip_ids else config.TEST_CLIPS
    if wanted:
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise ValueError("Test klibi bulunamadı: {}. Mevcut: {}".format(missing, sorted(by_id)))
        return [by_id[c] for c in wanted]
    chosen, seen = [], set()
    for c in sorted(test, key=lambda c: c["clip_id"]):
        if c["original_id"] not in seen:
            chosen.append(c)
            seen.add(c["original_id"])
        if len(chosen) == n:
            break
    return chosen


# ---------------------------------------------------------------------------
# IMU (stage1 + stage2)
# ---------------------------------------------------------------------------
def _butter(data, cutoff, fs, btype, order=config.IMU_FILTER_ORDER):
    b, a = butter(order, cutoff / (0.5 * fs), btype=btype, analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        padlen = len(data) - 1
    return filtfilt(b, a, data, padlen=padlen)


def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı: {}".format(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def imu_features(sensor_path, n_frames, fps, start_s=None):
    """
    IMU'yu video karelerine hizalar (lineer interpolasyon) ve yüksek geçiren filtreyle jitter çıkarır.
    Çıktı: (n_frames, 3) -> jitter_x, jitter_y, jitter_z
    """
    df = pd.read_csv(sensor_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.groupby("timestamp_ms", as_index=False).mean()   # aynı ms'deki tekrarları ortala (stage1)

    if "time_s" in df.columns and start_s is not None:
        # IDA_split: time_s kaydın başından itibaren; klibin başlangıcı manifest'teki start_s
        t_imu = df["time_s"].values - start_s
    else:
        t_imu = (df["timestamp_ms"].values - df["timestamp_ms"].values[0]) / 1000.0
    # IMU-video gecikmesi: IMU olayları videoda ~IMU_TIME_OFFSET_S önce görünür -> IMU zamanı geri çekilir
    t_imu = t_imu - float(getattr(config, "IMU_TIME_OFFSET_S", 0.0))

    t_video = np.arange(n_frames) / fps
    raw = df[["x", "y", "z"]].values
    synced = np.stack([
        interp1d(t_imu, raw[:, k], kind="linear", fill_value="extrapolate")(t_video) for k in range(3)
    ], axis=1)
    synced = to_portrait(synced)

    mode = config.IMU_FEATURE
    if mode in ("detrend_tilt", "tilt_motion"):
        # Telefon dikey (yerçekimi +y). Eğim açıları yerçekimi yönünden: roll = görüntü düzleminde dönme (yalpa)
        roll = np.arctan2(synced[:, 0], synced[:, 1])
        pitch = np.arctan2(synced[:, 2], synced[:, 1])
        feats = np.concatenate([synced, roll[:, None], pitch[:, None]], axis=1)
        out = imu_band(feats, fps, "detrend")                      # sarsıntı bandı (dx, dy, dθ için)
        if mode == "tilt_motion":
            # Kasıtlı hareket bandı (alpha için): yavaş eğim değişim hızı ve yavaş doğrusal ivme
            slow = gaussian_filter1d(feats, sigma=config.TRAJ_SIGMA, axis=0)
            roll_rate = np.gradient(slow[:, 3]) * fps                # rad/s
            pitch_rate = np.gradient(slow[:, 4]) * fps
            lin_acc = np.linalg.norm(slow[:, :3], axis=1) - np.linalg.norm(slow[:, :3], axis=1).mean()
            out = np.concatenate([out, roll_rate[:, None], pitch_rate[:, None], lin_acc[:, None]], axis=1)
        return out.astype(np.float32)
    return imu_band(synced, fps).astype(np.float32)


def to_portrait(acc, mode=None):
    """IMU eksenlerini IDA'nın dikey (portrait) düzenine çevirir: x' = görüntü sağı, y' = görüntü yukarısı (yerçekimi +y).
    Yatay (landscape) çekimde yerçekimi x ekseninde görünür:
      yerçekimi +x (telefonun üstü solda):  x' = -y, y' = x
      yerçekimi -x (telefonun üstü sağda):  x' =  y, y' = -x
    (Yürüyüş videosunda doğrulandı: bu dönüşümle IMU roll'u görüntüdeki dönmeyle IDA'daki gibi pozitif korelasyonlu.)"""
    mode = mode or getattr(config, "IMU_ORIENTATION", "auto")
    if mode == "portrait":
        return acc
    g = np.median(acc, axis=0)
    if mode == "auto" and abs(g[0]) <= abs(g[1]):
        return acc                                        # zaten dikey (IDA)
    if (mode == "auto" and g[0] > 0) or mode == "landscape_left":
        return np.stack([-acc[:, 1], acc[:, 0], acc[:, 2]], axis=1)
    return np.stack([acc[:, 1], -acc[:, 0], acc[:, 2]], axis=1)


def imu_channels(mode=None):
    """IMU_FEATURE'a göre model girdi kanalı sayısı."""
    return {"detrend_tilt": 5, "tilt_motion": 8}.get(mode or config.IMU_FEATURE, 3)


def imu_band(synced, fps, mode=None):
    """
    IMU'dan sarsıntı bandını çıkarır.

    "detrend" (varsayılan): IMU - Gaussian(IMU, sigma=TRAJ_SIGMA). Görsel hedefle (jitter = yörünge - Gaussian
        yörünge) BİREBİR aynı frekans bandı. IDA kayıtlarında görsel sarsıntının ~%100'ü 2 Hz'in altında
        (tepe ~0.6-0.8 Hz; tekne yalpası). Eski 2 Hz high-pass bu bandı tamamen siliyordu -> model hiçbir şey
        öğrenemiyordu (IMU ile görsel jitter korelasyonu ~0.00).
    "highpass": eski stage2 davranışı (Butterworth > IMU_CUTOFF_HZ). Karşılaştırma için bırakıldı.
    """
    mode = mode or config.IMU_FEATURE
    if mode in ("detrend", "detrend_tilt"):
        return synced - gaussian_filter1d(synced, sigma=config.TRAJ_SIGMA, axis=0)
    if mode == "highpass":
        return np.stack([_butter(synced[:, k], config.IMU_CUTOFF_HZ, fps, "high") for k in range(3)], axis=1)
    raise ValueError("Bilinmeyen IMU_FEATURE: {}".format(mode))


# ---------------------------------------------------------------------------
# Görsel ground truth (stage4)
# ---------------------------------------------------------------------------
def target_motion(data):
    """Önbellekten hedef hareketi seçer: (N-1, 3) tx, ty, dθ  (config.TARGET = "similarity")
    ya da eski davranış (N-1, 2) medyan dx, dy  (config.TARGET = "median")."""
    m = data["motion"]
    if config.TARGET == "median":
        return m[:, 0:2]
    return m[:, 2:5]


def model_unit(data_or_wh):
    """Modelin dx, dy birimi (orijinal piksel cinsinden): kısa kenar = FLOW_WIDTH olacak ölçekte 1 px.
    IDA (dikey, 480 px genişlik) için 480/320 = 1.5 px = RAFT akış birimi (değişiklik yok). Yatay videolarda da
    aynı açısal hareket aynı sayıya karşılık gelir (dikey/yatay çekim karışık eğitilebilir)."""
    if isinstance(data_or_wh, dict):
        W, H = int(data_or_wh["width"]), int(data_or_wh["height"])
    else:
        W, H = data_or_wh
    return min(W, H) / float(config.FLOW_WIDTH)


def to_model_units(motion, data):
    """RAFT akış birimindeki (flow_w genişlik) öteleme -> model birimi. dθ (rad) değişmez."""
    m = np.array(motion, dtype=np.float64, copy=True)
    u = model_unit(data)
    m[:, 0] *= (int(data["width"]) / float(int(data["flow_w"]))) / u
    m[:, 1] *= (int(data["height"]) / float(int(data["flow_h"]))) / u
    return m


def half_diag(data):
    """Model biriminde yarım köşegen: dönmeyi (rad) köşedeki kaymaya (model px) çevirmek için."""
    return 0.5 * float(np.hypot(int(data["width"]), int(data["height"]))) / model_unit(data)


def pan_alpha(speed):
    """
    Kasıtlı hareket (pan / dönüş) algılamaya dayalı alpha:
        alpha = 1 / (1 + (v / v0)^p),   v = yumuşatılmış kamera yolunun hızı (px/kare, dönme köşe px'ine çevrili)
    Kamera sabitken (v << v0) alpha -> 1 (tam stabilizasyon); kasıtlı pan sırasında (v >> v0) alpha -> 0
    (hareketle savaşma, gereksiz kırpma/bozulma yapma). Titreşimsiz geçiş için zamanda yumuşatılır.
    """
    a = 1.0 / (1.0 + (np.asarray(speed) / config.ALPHA_PAN_V0) ** config.ALPHA_PAN_POWER)
    a = gaussian_filter1d(a, sigma=config.ALPHA_SMOOTH_SIGMA)
    return np.clip(a, config.ALPHA_MIN, 1.0)


def optimal_alpha(gt, pred, rot_scale, sigma=None, shrink=None):
    """
    Öğrenilmiş düzeltme güveni hedefi: kalan sarsıntı |g - a p|^2'yi yerel olarak en aza indiren a (0..1).
        a*(t) = (G_s * Σ_c w_c^2 g_c p_c)(t) + λ  /  (G_s * Σ_c w_c^2 p_c^2)(t) + λ
    G_s: σ = ALPHA_SMOOTH_SIGMA kare Gaussian pencere, w: dθ için rot_scale (köşe px), λ: tahmin küçükken a*'ı 1'e
    çeker (düzeltme etkisizse tam güç). gt, pred: (N, D) aynı bileşen düzeninde (tahmin edilmeyen bileşen 0).
    """
    sigma = config.ALPHA_SMOOTH_SIGMA if sigma is None else sigma
    shrink = getattr(config, "ALPHA_OPT_SHRINK", 0.05) if shrink is None else shrink
    n = min(len(gt), len(pred))
    g, p = np.asarray(gt[:n], np.float64), np.asarray(pred[:n], np.float64)
    w2 = np.ones(g.shape[1])
    if g.shape[1] > 2:
        w2[2] = float(rot_scale) ** 2
    num = gaussian_filter1d((g * p * w2).sum(1), sigma)
    den = gaussian_filter1d((p * p * w2).sum(1), sigma)
    lam = shrink * float(den.mean()) + 1e-12
    a = (num + lam) / (den + lam)
    return np.clip(a, config.ALPHA_MIN, 1.0).astype(np.float32)


def visual_targets(motion, sigma=None, rot_scale=1.0):
    """
    motion: (N-1, D) kare i -> i+1 hareketi (D=2: dx, dy | D=3: dx, dy, dθ).
    Kare 0'ın konumu 0 olacak şekilde yörünge N kareye hizalanır, Gaussian(σ) ile yumuşatılır,
    jitter = yörünge - yumuşak yörünge. alpha hesabında dönme, rot_scale (yarım köşegen px) ile piksele çevrilir.
    """
    sigma = config.TRAJ_SIGMA if sigma is None else sigma
    trajectory = np.vstack([np.zeros((1, motion.shape[1])), np.cumsum(motion, axis=0)])
    smooth = gaussian_filter1d(trajectory, sigma=sigma, axis=0)
    jitter = trajectory - smooth

    w = np.ones(motion.shape[1])
    if motion.shape[1] > 2:
        w[2] = rot_scale
    smooth_motion = np.diff(smooth, axis=0, prepend=smooth[0:1])
    mot_mag = np.linalg.norm(smooth_motion * w, axis=1)          # kasıtlı hareket hızı (px/kare)
    if config.ALPHA_MODE in ("pan", "optimal"):
        # "optimal" hedefi modelin tahminine bağlıdır ve eğitimde (train.py) hesaplanır; burada yer tutucu pan alpha
        alpha = pan_alpha(mot_mag)
    else:   # "ratio": ilk tanım
        jit_mag = np.linalg.norm(jitter * w, axis=1)
        alpha = np.clip(jit_mag / (jit_mag + mot_mag + config.ALPHA_EPS), 0.0, 1.0)
    return {
        "trajectory": trajectory.astype(np.float32),
        "smooth": smooth.astype(np.float32),
        "jitter": jitter.astype(np.float32),
        "alpha": alpha.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Önbellek
# ---------------------------------------------------------------------------
def cache_path(clip_id):
    return os.path.join(config.CACHE_DIR, clip_id + ".npz")


def prepare_clip(clip, raft=None, need_gt=True, force=False):
    """
    Bir klip için IMU özelliklerini (+ istenirse RAFT ground truth'u) hesaplar, önbelleğe yazar ve döndürür.
    raft: RaftGlobalMotion örneği (need_gt=True ve önbellek yoksa gerekli).
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = cache_path(clip["clip_id"])
    data = {}
    if os.path.exists(path) and not force:
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        if int(data.get("version", -1)) != CACHE_VERSION:
            data = {}

    if "imu_jitter" not in data or str(data.get("imu_tag", "")) != imu_tag():
        info = video_info(clip["video"])
        data.update({
            "version": np.array(CACHE_VERSION),
            "fps": np.array(info["fps"]),
            "width": np.array(info["width"]),
            "height": np.array(info["height"]),
            "imu_jitter": imu_features(clip["sensor"], info["n_frames"], info["fps"], clip.get("start_s")),
            "imu_feature": np.array(config.IMU_FEATURE),
            "imu_tag": np.array(imu_tag()),
        })

    if need_gt and "motion" not in data:
        if raft is None:
            raise RuntimeError("RAFT gerekli ama verilmedi.")
        motion, finfo = raft.video_global_shifts(clip["video"])
        data["motion"] = motion            # (N-1, 6): med_dx, med_dy, tx, ty, dθ, dlog(s)
        data["shifts"] = motion[:, :2]     # eski medyan öteleme (karşılaştırma için)
        data["flow_w"] = np.array(finfo["flow_w"])
        data["flow_h"] = np.array(finfo["flow_h"])
        data["raft_bf16"] = np.array(int(getattr(raft, "bf16", False)))

    if "motion" in data:
        n = min(len(data["imu_jitter"]), len(data["motion"]) + 1)
        data["imu_jitter"] = data["imu_jitter"][:n]
        data["motion"] = data["motion"][:n - 1]
        data["shifts"] = data["motion"][:, :2]
        # Hedefler model biriminde (dikey/yatay videolar aynı ölçekte; IDA için birebir aynı)
        data.update(visual_targets(to_model_units(target_motion(data), data), rot_scale=half_diag(data)))

    np.savez_compressed(path, **data)
    return data


def has_gt_cache(clip):
    path = cache_path(clip["clip_id"])
    if not os.path.exists(path):
        return False
    with np.load(path) as z:
        return "motion" in z.files and int(z["version"]) == CACHE_VERSION


# ---------------------------------------------------------------------------
# Pencereleme (stage6)
# ---------------------------------------------------------------------------
def make_windows(imu_jitter, jitter, alpha, window=config.WINDOW_SIZE, stride=config.STRIDE):
    X, Y = [], []
    n = len(imu_jitter)
    target = np.concatenate([jitter, alpha[:, None]], axis=1)
    if n < window:
        return (np.zeros((0, window, imu_jitter.shape[1]), np.float32),
                np.zeros((0, window, target.shape[1]), np.float32))
    for i in range(0, n - window + 1, stride):
        X.append(imu_jitter[i:i + window])
        Y.append(target[i:i + window])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


COMPONENT_INDEX = {"dx": 0, "dy": 1, "dtheta": 2}


def imu_tag():
    """IMU özelliklerini belirleyen ayarlar; değişirse önbellekteki IMU özellikleri yeniden hesaplanır (RAFT değil)."""
    return "{}|off={}|or={}".format(config.IMU_FEATURE, float(getattr(config, "IMU_TIME_OFFSET_S", 0.0)),
                                    getattr(config, "IMU_ORIENTATION", "auto"))


def motion_components():
    """Modelin tahmin ettiği hareket bileşenleri (config.MOTION_COMPONENTS; median hedefinde dθ yok)."""
    comps = tuple(getattr(config, "MOTION_COMPONENTS", ("dx", "dy", "dtheta")))
    if config.TARGET == "median":
        comps = tuple(c for c in comps if c != "dtheta")
    return comps


def motion_index(comps=None):
    """Hedef jitter (dx, dy, dθ) sütunlarından modelin kullandıkları."""
    return [COMPONENT_INDEX[c] for c in (comps or motion_components())]


def motion_dim():
    """Model hareket çıktısı boyutu (örn. dy, dθ -> 2)."""
    return len(motion_components())


def smooth_speed(data):
    """Yumuşatılmış kamera yolunun hızı (px/kare; dönme köşe px'ine çevrili) — kasıtlı hareket ölçüsü."""
    sm = data["smooth"]
    w = np.ones(sm.shape[1])
    if sm.shape[1] > 2:
        w[2] = half_diag(data)
    return np.linalg.norm(np.diff(sm, axis=0, prepend=sm[0:1]) * w, axis=1)
