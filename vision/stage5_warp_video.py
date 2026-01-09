import cv2
import numpy as np
import os


def get_file_paths():
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    trajectory_path = os.path.join(project_root, "data", "videos", "visual_trajectory.npy")
    output_video_path = os.path.join(project_root, "outputs", "ground_truth_stabilization.mp4")

    return video_path, trajectory_path, output_video_path


def warp_video_ground_truth():
    print(f"--- STAGE 5: Ground Truth Stabilizasyon (Doğrulama) ---")
    video_path, trajectory_path, output_video_path = get_file_paths()

    # 1. Veri Kontrolü
    if not os.path.exists(video_path):
        print("❌ HATA: Video dosyası yok.")
        return
    if not os.path.exists(trajectory_path):
        print("❌ HATA: Yörünge verisi (visual_trajectory.npy) yok. Stage 4'ü çalıştır.")
        return

    # Yörünge verisini yükle
    # [dx, dy, traj_x, traj_y, smooth_x, smooth_y, jitter_x, jitter_y]
    data = np.load(trajectory_path)
    # Bize lazım olan: (smooth - trajectory) yani -jitter
    # Ama dikkat: Stage 4'te hesaplanan veriler "Küçültülmüş" (320px) çözünürlükteydi.
    # Orijinal video ise muhtemelen 1080p veya 480p.
    # Bu yüzden bir ÖLÇEKLEME (Scaling) yapmamız gerekecek.

    smooth_path = data[:, 4:6]
    orig_path = data[:, 2:4]
    # Düzeltme Vektörü = Hedef (Smooth) - Mevcut (Orijinal)
    correction_vectors = smooth_path - orig_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- ÖLÇEK FAKTÖRÜ HESABI ---
    # Stage 3'te kullandığımız referans genişlik 320 idi.
    REF_WIDTH = 320.0

    # Eğer orijinal video 320'den farklıysa düzeltme vektörlerini büyütmeliyiz.
    # Not: RAFT kısmında 8'in katına yuvarlama yapmıştık, ama oran kabaca width/320'dir.
    # Hassas hesap için Stage 3'teki mantığın aynısını kuralım:
    resize_ratio = REF_WIDTH / width
    # scale_factor = Orijinal / Küçültülmüş
    scale_factor = 1.0 / resize_ratio

    print(f"📏 Orijinal Boyut: {width}x{height}")
    print(f"🔍 Flow Referans Genişliği: {REF_WIDTH}")
    print(f"✖️  Ölçek Çarpanı: {scale_factor:.4f}")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("🚀 Video stabilize ediliyor (Warping)...")

    # Siyah kenarları engellemek için ne kadar 'crop' (kırpma) yapacağımız opsiyoneldir.
    # Şimdilik kırpma yapmadan siyah kenarları görelim ki stabilizasyonu anlayalım.

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        if i < len(correction_vectors):
            # O anki kare için düzeltme vektörü (dx, dy)
            dx = correction_vectors[i, 0] * scale_factor
            dy = correction_vectors[i, 1] * scale_factor

            # Affine Dönüşüm Matrisi (2x3)
            # [ 1  0  dx ]
            # [ 0  1  dy ]
            M = np.float32([[1, 0, dx], [0, 1, dy]])

            # Görüntüyü kaydır
            stabilized_frame = cv2.warpAffine(frame, M, (width, height))

            out.write(stabilized_frame)
        else:
            # Vektör bittiyse (son kareler) olduğu gibi yaz
            out.write(frame)

        if (i + 1) % 30 == 0:
            print(f"   Kare: {i + 1}/{total_frames}")

    cap.release()
    out.release()
    print(f"✅ Ground Truth videosu oluşturuldu: {output_video_path}")
    print("   (Bu videoyu izleyin. Eğer stabilse, CNN eğitimi için hedef verimiz sağlam demektir.)")


if __name__ == "__main__":
    warp_video_ground_truth()