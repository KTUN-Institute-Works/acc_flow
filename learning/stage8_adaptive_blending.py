import torch
import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt

# Model dosyasını import edebilmek için
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

from models.imu_alpha_net import IMUStabilizerNet


def get_file_paths():
    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    imu_features_path = os.path.join(project_root, "data", "sensors", "processed_features.csv")
    model_path = os.path.join(project_root, "models", "best_imu_model.pth")
    output_video_path = os.path.join(project_root, "outputs", "final_cnn_stabilized.mp4")
    plot_path = os.path.join(project_root, "outputs", "final_comparison_plot.png")
    return video_path, imu_features_path, model_path, output_video_path, plot_path


def apply_adaptive_stabilization():
    start_time = time.time()
    print(f"--- STAGE 8: CNN Tabanlı Final Stabilizasyon ---")
    video_path, imu_path, model_path, out_path, plot_path = get_file_paths()

    # 1. Modeli Yükle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUStabilizerNet().to(device)

    if not os.path.exists(model_path):
        print("❌ HATA: Model dosyası yok.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("🧠 Model yüklendi ve hazır.")

    # 2. Tüm IMU Verisini Hazırla
    # Stage 6'da pencereleme yapmıştık, şimdi TÜM videoyu tek seferde tahmin edeceğiz.
    # Conv1D yapısı sayesinde model değişken uzunlukta girdi alabilir!
    df_imu = pd.read_csv(imu_path)

    # Sadece jitter verilerini al
    imu_data = df_imu[['jitter_x', 'jitter_y', 'jitter_z']].values

    # Tensor'a çevir: (1, Length, 3)
    input_tensor = torch.from_numpy(imu_data).float().unsqueeze(0).to(device)

    # 3. Tahmin (Inference)
    print("🔮 Tüm video için sarsıntı tahmini yapılıyor...")
    with torch.no_grad():
        # Çıktı: (1, Length, 2) -> (dx, dy)
        predicted_offsets = model(input_tensor)

    # Numpy'a geri dön
    offsets = predicted_offsets.squeeze(0).cpu().numpy()  # (Length, 2)

    print(f"✅ Tahmin tamamlandı. Shape: {offsets.shape}")

    # 4. Video İşleme (Warping)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale Factor (Stage 5'ten hatırladığımız)
    # Model 320px referansıyla eğitildi (çünkü Target verisi öyleydi).
    # Orijinal video için bunu büyütmeliyiz.
    REF_WIDTH = 320.0
    processed_width = 320.0  # RAFT scale
    scale_factor = W / processed_width

    # Adaptive Blending Parametresi (Alpha)
    # CNN çıktısına ne kadar güveniyoruz?
    # 1.0 = Tamamen CNN ne derse o (Rijit).
    # 0.8 = Biraz yumuşat.
    ALPHA = 1.0


    # --- GLOBAL CROP HESABI ---
    scaled_dx = -(offsets[:, 0] * scale_factor * ALPHA)
    scaled_dy = -(offsets[:, 1] * scale_factor * ALPHA)

    min_dx = np.min(scaled_dx)
    max_dx = np.max(scaled_dx)

    min_dy = np.min(scaled_dy)
    max_dy = np.max(scaled_dy)

    # valid region
    x1 = max(0, int(np.ceil(max_dx)))
    x2 = min(W, int(np.floor(W + min_dx)))

    y1 = max(0, int(np.ceil(max_dy)))
    y2 = min(H, int(np.floor(H + min_dy)))

    print("Crop region:", x1, y1, x2, y2)

    # Side-by-Side Video
    crop_W = x2 - x1
    crop_H = y2 - y1

    if crop_W <= 0 or crop_H <= 0:
        print("❌ HATALI CROP:", crop_W, crop_H)
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_orig = cv2.VideoWriter(os.path.join(project_root, "outputs", "original_crop.mp4"), fourcc, fps,
                               (crop_W, crop_H))

    out_stab = cv2.VideoWriter(os.path.join(project_root, "outputs", "stabilized_crop.mp4"), fourcc, fps,
                               (crop_W, crop_H))

    out_overlay = cv2.VideoWriter(os.path.join(project_root, "outputs", "overlay.mp4"), fourcc, fps, (crop_W, crop_H))


    # 3 panel (aralarda çizgi var)
    panel_width = crop_W * 3 + 10  # 2 çizgi * 5px
    out_panel = cv2.VideoWriter(os.path.join(project_root, "outputs", "compare_3panel.mp4"), fourcc, fps,
                                (panel_width, crop_H))
    out_final = cv2.VideoWriter(out_path, fourcc, fps, (crop_W, crop_H))
    font = cv2.FONT_HERSHEY_SIMPLEX

    trajectory_x = []
    trajectory_y = []



    print("🚀 Video render ediliyor...")
    max_frames = min(total_frames, len(offsets))
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break

        frame_orig = frame.copy()
        frame_orig = frame_orig[y1:y2, x1:x2]
        frame_stab = frame.copy()

        # Eğer tahmin verimiz varsa uygula
        if i < len(offsets):
            # Model çıktısı: Tahmin edilen Jitter
            # Stabilizasyon için bu jitter'ı TERS yönde uygulamalıyız (-1 ile çarp)
            # Stage 5'te Method A (Smooth - Orig) doğru çıkmıştı.
            # Yani Jitter pozitifsie görüntü sağa kaymıştır, biz sola (-dx) itmeliyiz.

            pred_dx = offsets[i, 0]
            pred_dy = offsets[i, 1]

            # İstatistik için kaydet
            trajectory_x.append(pred_dx)
            trajectory_y.append(pred_dy)

            # Ölçekle ve Uygula
            dx = -(pred_dx * scale_factor * ALPHA)
            dy = -(pred_dy * scale_factor * ALPHA)

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame_stab = cv2.warpAffine(
                frame,
                M,
                (W, H),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        frame_stab = frame_stab[y1:y2, x1:x2]

        # --- OVERLAY ---
        BETA = 0.5
        overlay = cv2.addWeighted(frame_orig, 1 - BETA, frame_stab, BETA, 0)

        # --- LABELS ---
        cv2.putText(frame_orig, "Original", (20, 40), font, 1.0, (0, 0, 255), 2)
        cv2.putText(frame_stab, "", (20, 40), font, 1.0, (0, 255, 0), 2)
        cv2.putText(overlay, "Overlay", (20, 40), font, 1.0, (255, 255, 255), 2)

        # --- 3 PANEL ---
        line = np.ones((crop_H, 5, 3), dtype=np.uint8) * 255

        panel = np.hstack([
            frame_orig,
            line,
            frame_stab,
            line,
            overlay
        ])

        out_orig.write(frame_orig)
        out_stab.write(frame_stab)
        out_overlay.write(overlay)
        out_panel.write(panel)
        out_final.write(frame_stab)

        if (i + 1) % 20 == 0:
            print(f"   Kare: {i + 1}/{total_frames}")

    cap.release()
    out_orig.release()
    out_stab.release()
    out_overlay.release()
    out_panel.release()
    out_final.release()
    end_time = time.time()
    print(f"✅ Final Video Hazır: {out_path}")
    print(f"⏰ Toplam süre: {end_time - start_time:.2f} saniye")

    # 5. Sonuç Grafiği (Makale için harika bir görsel)
    plt.figure(figsize=(12, 6))
    plt.plot(trajectory_x, label='CNN Predicted Jitter (X)', color='green')
    plt.plot(trajectory_y, label='CNN Predicted Jitter (Y)', color='blue', alpha=0.6)
    plt.title(f"CNN Tarafından İvmeölçerden Tahmin Edilen Görsel Kayma")
    plt.xlabel("Frame")
    plt.ylabel("Predicted Pixel Shift (320px base)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"📊 Final grafik kaydedildi: {plot_path}")


if __name__ == "__main__":
    apply_adaptive_stabilization()