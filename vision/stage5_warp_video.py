import cv2
import numpy as np
import os


def get_file_paths():
    """
    Stage 5 (Debug) işlemi için dosya yollarını üretir.

    Döndürdüğü yollar:
    - video_path: İşlem görecek orijinal ham video.
    - trajectory_path: Stage 4'te hesaplanan yörünge verileri (numpy matrisi).
    - output_video_path: 4 farklı stabilizasyon denemesinin birleştirildiği 2x2 grid çıktı videosu.
    """
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    trajectory_path = os.path.join(project_root, "data", "videos", "visual_trajectory.npy")
    output_video_path = os.path.join(project_root, "outputs", "debug_stabilization.mp4")

    return video_path, trajectory_path, output_video_path


def warp_video_debug():
    """
    Hesaplanan kamera yörüngelerinin videoya uygulanması sırasında oluşabilecek
    yön (işaret) ve ölçek (scale) hatalarını görsel olarak tespit etmek için
    2x2 ızgara (grid) formatında bir test videosu oluşturur.

    İşlem Adımları:
    1. Yörünge verisi yüklenir ve optik akış boyutu (320px) ile orijinal video
       boyutu arasındaki ölçek faktörü (scale_factor) hesaplanır.
    2. Her kare için 3 farklı stabilizasyon hipotezi (Affine Transformation) uygulanır:
       - Method A: Standart stabilizasyon (Smooth - Orig).
       - Method B: Ters yönlü stabilizasyon (Orig - Smooth).
       - Method C: Yarı güçlü stabilizasyon (Method A * 0.5).
    3. Orijinal kare ve bu 3 hipotez aynı ekranda birleştirilip yeni bir MP4 olarak kaydedilir.
    """
    print(f"--- STAGE 5: Stabilizasyon Hata Ayıklama (Grid View) ---")
    video_path, trajectory_path, output_video_path = get_file_paths()

    # 1. Veri Yükleme
    if not os.path.exists(video_path) or not os.path.exists(trajectory_path):
        print("HATA: Dosyalar eksik.")
        return

    data = np.load(trajectory_path)
    orig_path = data[:, 2:4]
    smooth_path = data[:, 4:6]

    # İki farklı hipotez
    # Hipotez A: Mevcut yöntem (Smooth - Orig)
    diff_A = smooth_path - orig_path

    # Hipotez B: Tam tersi (Orig - Smooth)
    diff_B = orig_path - smooth_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ölçekleme (RAFT 320px -> Orijinal Boyut)
    # Stage 3'te RAFT için 8'e yuvarlamıştık (320 -> 320)
    # Ama eğer video boyutu farklıysa scale önemli.
    # En güvenli yöntem: Manuel oran.
    REF_WIDTH = 320.0
    # RAFT kodu '((320 // 8) * 8)' yapmıştı. Genelde 320 çıkar.
    processed_width = 320.0
    scale_factor = W / processed_width

    print(f"Orijinal: {W}x{H} | Scale: {scale_factor:.2f}")

    # Grid Video (2x Genişlik, 2x Yükseklik)
    out_W = W * 2
    out_H = H * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_W, out_H))

    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Debug videosu hazırlanıyor...")

    for i in range(frames):
        ret, frame = cap.read()
        if not ret: break

        # Kareleri hazırla
        frame_orig = frame.copy()
        frame_A = frame.copy()  # Smooth - Orig
        frame_B = frame.copy()  # Orig - Smooth (Inverted)
        frame_C = frame.copy()  # Half Scale (Scale hatası kontrolü)

        if i < len(diff_A):
            # --- YÖNTEM A ---
            dx_a = diff_A[i, 0] * scale_factor
            dy_a = diff_A[i, 1] * scale_factor
            M_a = np.float32([[1, 0, dx_a], [0, 1, dy_a]])
            frame_A = cv2.warpAffine(frame, M_a, (W, H))

            # --- YÖNTEM B (TERS İŞARET) ---
            dx_b = diff_B[i, 0] * scale_factor
            dy_b = diff_B[i, 1] * scale_factor
            M_b = np.float32([[1, 0, dx_b], [0, 1, dy_b]])
            frame_B = cv2.warpAffine(frame, M_b, (W, H))

            # --- YÖNTEM C (YARI GÜÇ) ---
            # Belki scale çok büyüktür?
            dx_c = dx_a * 0.5
            dy_c = dy_a * 0.5
            M_c = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
            frame_C = cv2.warpAffine(frame, M_c, (W, H))

        # Etiketler
        cv2.putText(frame_orig, "ORIGINAL", (30, 80), font, 2, (0, 0, 255), 4)
        cv2.putText(frame_A, "Method A (Smooth-Orig)", (30, 80), font, 2, (0, 255, 0), 4)
        cv2.putText(frame_B, "Method B (Orig-Smooth)", (30, 80), font, 2, (0, 255, 255), 4)
        cv2.putText(frame_C, "Method C (Half Scale)", (30, 80), font, 2, (255, 0, 255), 4)

        # Birleştirme (2x2 Grid)
        # [ Orig   | Method A ]
        # [ Method B | Method C ]
        top_row = np.hstack([frame_orig, frame_A])
        bottom_row = np.hstack([frame_B, frame_C])
        grid = np.vstack([top_row, bottom_row])

        # Grid çok büyük olabilir, izlemek için küçültelim (Opsiyonel)
        # grid_resized = cv2.resize(grid, (W, H)) # Tek ekran boyutuna sığdır
        # out.write(grid_resized)
        out.write(grid)

        if (i + 1) % 20 == 0:
            print(f"   Kare: {i + 1}/{frames}")

    cap.release()
    out.release()
    print(f"\nDebug videosu hazır: {output_video_path}")
    print("Videoyu izleyin. Hangi kare stabil duruyor?")
    print("   - Method A iyiyse: Formül doğru, scale veya smoothing ayarı yanlış.")
    print("   - Method B iyiyse: İşaret hatası yapmışız (Ters çevireceğiz).")
    print("   - Method C iyiyse: Scale factor çok büyük gelmiş.")


if __name__ == "__main__":
    warp_video_debug()