import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Proje kök dizinini yola ekle
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)


# --- METRİK HESAPLAMA FONKSİYONLARI ---

def calculate_distortion(H):
    """B_t matrisinden anizotropik ölçekleme (Distortion) hesaplar[cite: 964, 965]."""
    affine = H[0:2, 0:2]
    _, s, _ = np.linalg.svd(affine)
    # Literatürde s_min / s_max oranı 1.0'a ne kadar yakınsa o kadar az bozulma demektir[cite: 801, 964].
    return s[1] / s[0] if s[0] != 0 else 0


def calculate_crop_ratio(H, width, height):
    """B_t matrisine göre görüntüde kalan geçerli alan oranını hesaplar[cite: 959, 962]."""
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).T
    transformed_corners = np.dot(H, corners)
    transformed_corners /= transformed_corners[2]

    # Orijinal çerçeve ile kesişen overlap bölgesini bul [cite: 962]
    min_x = max(0, np.min(transformed_corners[0]))
    max_x = min(width, np.max(transformed_corners[0]))
    min_y = max(0, np.min(transformed_corners[1]))
    max_y = min(height, np.max(transformed_corners[1]))

    crop_area = max(0, max_x - min_x) * max(0, max_y - min_y)
    return crop_area / (width * height)


def calculate_stability(trajectories):
    """Hareket sinyallerinin düşük frekanslı enerji oranını hesaplar[cite: 967, 968]."""
    if len(trajectories) < 10: return 0
    fft_vals = np.abs(np.fft.fft(trajectories))
    fft_vals = fft_vals[1:len(fft_vals) // 2]  # DC bileşenini çıkar [cite: 967]

    # Literatürdeki 2. ile 6. frekanslar arası enerji oranı [cite: 967]
    low_freq_energy = np.sum(fft_vals[1:6] ** 2)
    total_energy = np.sum(fft_vals ** 2)

    return low_freq_energy / total_energy if total_energy != 0 else 0


# --- ÖZELLİK EŞLEŞTİRME VE HOMOGRAFİ ---

def get_homography(method, img1, img2):
    """Seçilen yönteme göre iki kare arasındaki düzeltme matrisini (B_t) döndürür[cite: 961, 1370]."""
    if method == 'KLT':
        # FAST/KLT kombinasyonu video için en verimli yoldur [cite: 718]
        p1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 500, 0.01, 10)
        if p1 is None: return None
        p2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, p1, None)
        p1, p2 = p1[status == 1], p2[status == 1]
    else:
        # SIFT veya ORB tabanlı özellik çıkarımı [cite: 903, 911]
        detector = cv2.SIFT_create() if method == 'SIFT' else cv2.ORB_create()
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None: return None

        norm = cv2.NORM_L2 if method == 'SIFT' else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4: return None

        p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    return H


# --- ANA ANALİZ DÖNGÜSÜ ---

def run_comprehensive_evaluation(orig_path, model_configs):
    """Tüm modelleri farklı algoritmalarla çapraz test eder."""
    final_rows = []
    tracking_methods = ['KLT', 'SIFT', 'ORB']

    for model_name, stab_path in model_configs.items():
        print(f"\n🚀 Analiz Ediliyor: {model_name}")

        for t_method in tracking_methods:
            print(f"  -> İzleyici: {t_method}")
            cap_orig = cv2.VideoCapture(orig_path)
            cap_stab = cv2.VideoCapture(stab_path)

            w_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

            stats = {'dist': [], 'crop': [], 'trans': []}
            start_t = time.time()
            f_count = 0

            while True:
                ret1, frame_orig = cap_orig.read()
                ret2, frame_stab = cap_stab.read()
                if not ret1 or not ret2: break

                # Farklı çözünürlükleri orijinal boyuta normalize et
                frame_stab_resized = cv2.resize(frame_stab, (w_orig, h_orig))

                try:
                    Bt = get_homography(t_method, frame_orig, frame_stab_resized)
                    if Bt is not None:
                        stats['dist'].append(calculate_distortion(Bt))
                        stats['crop'].append(calculate_crop_ratio(Bt, w_orig, h_orig))
                        # Translasyon sinyali: sqrt(dx^2 + dy^2) [cite: 797]
                        stats['trans'].append(np.sqrt(Bt[0, 2] ** 2 + Bt[1, 2] ** 2))
                        f_count += 1
                except Exception:
                    continue

            # Verimlilik ölçümü (FPS)
            elapsed = time.time() - start_t
            fps = f_count / elapsed if elapsed > 0 else 0

            cap_orig.release()
            cap_stab.release()

            final_rows.append({
                'Model': model_name,
                'Tracking': t_method,
                'Stability': calculate_stability(stats['trans']),
                'Distortion': np.mean(stats['dist']) if stats['dist'] else 0,
                'CropRatio': np.mean(stats['crop']) if stats['crop'] else 0,
                'AnalysisFPS': fps
            })

    return pd.DataFrame(final_rows)


# --- ÇALIŞTIRMA ---

if __name__ == "__main__":
    video_in = os.path.join(project_root, "data", "videos", "input.mp4")

    models = {
        'DIFRINT': os.path.join(project_root, "data", "videos", "DIFRINT_stable.mp4"),
        'StabNet': os.path.join(project_root, "data", "videos", "StabNet_stable.mp4"),
        'DUT': os.path.join(project_root, "data", "videos", "DUT_stable.mp4"),
        'NNDVS': os.path.join(project_root, "data", "videos", "Regular", "output", "0.mp4"),
        'Ours (IMU-CNN)': os.path.join(project_root, "outputs", "final_cnn_stabilized.mp4")
    }

    df_results = run_comprehensive_evaluation(video_in, models)

    # Pivot tablo ile teze hazır hale getirme
    print("\n" + "=" * 50)
    print("FİNAL KARŞILAŞTIRMA TABLOSU")
    print("=" * 50)
    pivot_table = df_results.pivot(index='Model', columns='Tracking', values=['Stability', 'Distortion', 'AnalysisFPS'])
    print(pivot_table)

    # CSV olarak kaydet
    df_results.to_csv(os.path.join(project_root, "outputs", "method_comparison_results.csv"), index=False)