import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d


def get_file_paths():
    """
    Stage 4 işlemi için dosya yollarını dinamik olarak belirler.

    Döndürdüğü yollar:
    - flow_path: Stage 3'te üretilen piksel bazlı optik akış verisi (optical_flow.npy).
    - output_path: Hesaplanan yörünge, yumuşatılmış yol ve görsel sarsıntı verilerinin
      birleştirilip kaydedileceği numpy dosyası (visual_trajectory.npy).
    - plot_path: Yörünge ve sarsıntı analizinin çizileceği grafik dosyası.
    """
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    flow_path = os.path.join(project_root, "data", "videos", "optical_flow.npy")
    output_path = os.path.join(project_root, "data", "videos", "visual_trajectory.npy")
    plot_path = os.path.join(project_root, "outputs", "visual_trajectory_analysis.png")

    return flow_path, output_path, plot_path


def compute_trajectory():
    """
    Optik akış verilerini kullanarak kameranın sanal yörüngesini hesaplar ve
    görsel sarsıntıyı (visual jitter) izole eder.

    İşlem Adımları:
    1. Optik akış verisi (U, V) yüklenir.
    2. Sahnede hareket eden nesneleri (foreground) göz ardı etmek ve arka planın
       (kameranın) hareketini bulmak için her karedeki vektörlerin Medyanı alınır.
    3. Elde edilen kareler arası kaymalar (shifts) kümülatif olarak toplanarak (cumsum)
       kameranın orijinal, titrek yörüngesi oluşturulur.
    4. 1D Gaussian Filtresi (Sigma=15) kullanılarak bu yörünge pürüzsüzleştirilir (smooth).
    5. Orijinal yörünge ile pürüzsüz yörünge arasındaki fark alınarak 'Görsel Jitter' bulunur.
    6. Kare kaymaları, orijinal yörünge, pürüzsüz yörünge ve jitter değerleri
       (N_frames x 8) boyutunda bir dizi halinde kaydedilir.
    """
    print(f"--- STAGE 4: Görsel Yörünge ve Smoothing ---")
    flow_path, output_path, plot_path = get_file_paths()

    if not os.path.exists(flow_path):
        print("HATA: optical_flow.npy bulunamadı. Stage 3'ü çalıştır.")
        return

    flow = np.load(flow_path)
    print(f"Flow Verisi Yüklendi: {flow.shape}")

    num_frames = flow.shape[0]

    global_shifts = []
    print("Global hareket hesaplanıyor (Median Flow)...")

    for i in range(num_frames):
        dx_map = flow[i, ..., 0]
        dy_map = flow[i, ..., 1]

        dx_global = np.median(dx_map)
        dy_global = np.median(dy_map)

        global_shifts.append([dx_global, dy_global])

    global_shifts = np.array(global_shifts)  # (N, 2)

    trajectory = np.cumsum(global_shifts, axis=0)

    SIGMA = 15
    smooth_trajectory = gaussian_filter1d(trajectory, sigma=SIGMA, axis=0)

    visual_jitter = trajectory - smooth_trajectory

    final_data = np.hstack([
        global_shifts,  # 0, 1
        trajectory,  # 2, 3
        smooth_trajectory,  # 4, 5
        visual_jitter  # 6, 7
    ])

    np.save(output_path, final_data)
    print(f"Görsel yörünge verisi kaydedildi: {output_path}")
    print(f"Shape: {final_data.shape} (N_frames x 8)")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.title(f"Kamera Yörüngesi (X Ekseni) - Sigma={SIGMA}")
    plt.plot(trajectory[:, 0], label='Orijinal (Titrek)', alpha=0.6, color='gray')
    plt.plot(smooth_trajectory[:, 0], label='Smooth (Hedef)', color='green', linewidth=2)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title("Görüntüden Elde Edilen Jitter (Görsel Sarsıntı)")
    plt.plot(visual_jitter[:, 0], label='Visual Jitter (X)', color='red')
    plt.plot(visual_jitter[:, 1], label='Visual Jitter (Y)', color='blue', alpha=0.5)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Analiz grafiği oluşturuldu: {plot_path}")


if __name__ == "__main__":
    compute_trajectory()