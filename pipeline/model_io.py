"""Model kaydetme / yükleme ve IMU normalizasyonu."""
import numpy as np
import torch

from models.imu_alpha_net import IMUAdaptiveStabilizerNet


def save_checkpoint(path, model, imu_mean, imu_std, extra=None):
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "imu_mean": np.asarray(imu_mean, np.float32).tolist(),
        "imu_std": np.asarray(imu_std, np.float32).tolist(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_model(path, device):
    """Yeni (dict) ve eski (sadece state_dict, normalizasyonsuz) checkpoint formatlarını destekler."""
    try:
        # Kendi eğitimimizin checkpoint'i (güvenilir kaynak). PyTorch >= 2.6 varsayılan weights_only=True,
        # checkpoint'teki meta veriler (ör. numpy skalerleri) yüzünden yüklemeyi reddedebiliyor.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:          # weights_only argümanı olmayan eski PyTorch
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model = IMUAdaptiveStabilizerNet(dilations=tuple(ckpt.get("dilations", (1, 1, 1, 1))),
                                         in_channels=int(ckpt.get("in_channels", 3)),
                                         motion_dim=int(ckpt.get("motion_dim", 2)))
        model.load_state_dict(ckpt["state_dict"])
        mean = np.asarray(ckpt.get("imu_mean", [0, 0, 0]), np.float32)
        std = np.asarray(ckpt.get("imu_std", [1, 1, 1]), np.float32)
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        model = IMUAdaptiveStabilizerNet()
        model.load_state_dict(ckpt)
        mean, std, meta = np.zeros(3, np.float32), np.ones(3, np.float32), {}
    model.to(device).eval()
    return model, mean, std, meta


@torch.no_grad()
def predict_sequence(model, imu_jitter, mean, std, device, target_std=1.0):
    """Tüm klibi tek seferde tahmin eder (tam evrişimli model değişken uzunluk kabul eder).
    Dönüş: offsets (N,D) [FLOW_WIDTH ölçeğinde dx, dy (px) ve varsa dθ (rad)], alpha (N,)
    target_std: eğitimde hedefler bu std'ye bölündü; çıktı geri ölçeklenir (eski modeller için 1.0)."""
    x = (np.asarray(imu_jitter, np.float32) - mean) / std
    t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    motion, alpha = model(t)
    offsets = motion.squeeze(0).cpu().numpy() * target_std
    alpha = np.clip(alpha.squeeze(0).squeeze(-1).cpu().numpy(), 0.0, 1.0)
    return offsets, alpha
