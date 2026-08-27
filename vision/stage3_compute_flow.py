import sys
import os
import cv2
import numpy as np
import torch
import argparse
from collections import OrderedDict


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raft_path = os.path.join(project_root, 'RAFT')

sys.path.append(raft_path)

try:
    from core.raft import RAFT
    from core.utils.utils import InputPadder
except ImportError:
    print("HATA: RAFT modülleri bulunamadı.")
    print(f"Aranan yol: {raft_path}")
    print("Lütfen RAFT klasörünün proje ana dizininde olduğundan emin olun.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_WEIGHTS = os.path.join(project_root, "models", "raft-things.pth")


class InputArgs:
    """
    RAFT modelinin konfigürasyon argümanlarını tutan yardımcı (wrapper) sınıf.
    RAFT orijinal kaynak kodu, argümanları hem dictionary ('dropout' in args)
    hem de attribute (args.small) mantığıyla kontrol ettiği için, argparse
    yerine bu melez yapı kullanılmıştır.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

    def __contains__(self, key):
        return key in self.__dict__


def get_file_paths():
    """
    Stage 3 işlemi için gerekli girdi ve çıktı dosya yollarını üretir.

    Döndürdüğü yollar:
    - video_path: İşlenecek kaynak video dosyası (input.mp4).
    - output_npy_path: Hesaplanan optik akış vektörlerinin (U, V) ham matematiksel formatta kaydedileceği dosya.
    - vis_video_path: Optik akışın renk kodlu (HSV) olarak görselleştirildiği çıktı videosu.
    """
    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    output_npy_path = os.path.join(project_root, "data", "videos", "optical_flow.npy")
    vis_video_path = os.path.join(project_root, "outputs", "optical_flow_raft.mp4")
    return video_path, output_npy_path, vis_video_path


def load_image(img_bgr):
    """
    OpenCV'nin (Height, Width, Channels) ve BGR formatındaki matrisini alır,
    RAFT'ın beklediği PyTorch Tensor formatına (Batch, Channels, Height, Width)
    ve RGB dizilimine dönüştürür.
    """
    img = torch.from_numpy(img_bgr).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def compute_flow_raft():
    """
    RAFT modelini kullanarak videodaki ardışık kareler arasındaki optik akışı (piksel hareketlerini) hesaplar.

    İşlem Adımları:
    1. Sistemde GPU/CPU durumu kontrol edilir ve RAFT önceden eğitilmiş ağırlıklarıyla (raft-things.pth) yüklenir.
    2. Video okunur ve bellek/hız optimizasyonu için çözünürlüğü küçültülerek 8'in katlarına yuvarlanır.
    3. Her ardışık iki kare (Frame t ve Frame t+1) modele sokularak piksel bazlı hareket (U, V vektörleri) tahmin edilir.
    4. Bu vektörler görsel analiz için yönün renge, hızın parlaklığa dönüştüğü bir HSV videoya dönüştürülür.
    5. Yapay zeka pipeline'ının bir sonraki adımında kullanılmak üzere ham tensor verisi NumPy (.npy) olarak kaydedilir.
    """
    # ...
    print(f"--- STAGE 3: Optical Flow Hesaplama (RAFT) ---")
    print(f"Cihaz: {DEVICE}")
    if DEVICE == 'cpu':
        print("UYARI: CPU üzerinde RAFT çok yavaş çalışabilir.")

    video_path, output_npy_path, vis_video_path = get_file_paths()

    # 1. KONTROLLER
    if not os.path.exists(video_path):
        print(f"HATA: Video dosyası bulunamadı: {video_path}")
        return
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"HATA: Model ağırlık dosyası bulunamadı: {MODEL_WEIGHTS}")
        return

    args = InputArgs(
        model=MODEL_WEIGHTS,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0.0
    )

    model = RAFT(args)

    if DEVICE == 'cpu':
        state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(args.model)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    print("RAFT modeli başarıyla yüklendi.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    RESIZE_WIDTH = 320
    ratio = RESIZE_WIDTH / width
    RESIZE_HEIGHT = int(height * ratio)

    RESIZE_WIDTH = ((RESIZE_WIDTH // 8) * 8)
    RESIZE_HEIGHT = ((RESIZE_HEIGHT // 8) * 8)

    print(f"İşlem Boyutu: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vis = cv2.VideoWriter(vis_video_path, fourcc, fps, (RESIZE_WIDTH, RESIZE_HEIGHT))

    ret, prev_frame = cap.read()
    if not ret:
        print("HATA: Video okunamadı.")
        return

    prev_frame = cv2.resize(prev_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    flow_data = []

    print("Akış hesaplanıyor...")

    with torch.no_grad():
        for i in range(total_frames - 1):
            ret, curr_frame = cap.read()
            if not ret: break

            curr_frame = cv2.resize(curr_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            image1 = load_image(prev_frame)
            image2 = load_image(curr_frame)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # RAFT Tahmini
            flow_low, flow_up = model(image1, image2, iters=10,
                                      test_mode=True)  # CPU için iters 20->10 düşürdüm hız için

            flow_up = padder.unpad(flow_up)
            flow_numpy = flow_up[0].permute(1, 2, 0).cpu().numpy()

            flow_data.append(flow_numpy)

            # Görselleştirme
            mag, ang = cv2.cartToPolar(flow_numpy[..., 0], flow_numpy[..., 1])
            hsv = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            out_vis.write(rgb_flow)
            prev_frame = curr_frame

            # İlerleme çubuğu
            if (i + 1) % 5 == 0:
                print(f"   Kare: {i + 1}/{total_frames}")

    cap.release()
    out_vis.release()

    final_flow = np.array(flow_data, dtype=np.float32)
    np.save(output_npy_path, final_flow)

    print(f"\nRAFT Flow verisi kaydedildi: {output_npy_path}")
    print(f"Boyut: {final_flow.shape}")
    print(f"Önizleme videosu: {vis_video_path}")


if __name__ == "__main__":
    compute_flow_raft()