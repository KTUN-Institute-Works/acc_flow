import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d


def get_file_paths():
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    flow_path = os.path.join(project_root, "data", "videos", "optical_flow.npy")
    output_path = os.path.join(project_root, "data", "videos", "visual_trajectory.npy")
    plot_path = os.path.join(project_root, "outputs", "visual_trajectory_analysis.png")

    return flow_path, output_path, plot_path


def compute_trajectory():
    print(f"--- STAGE 4: Görsel Yörünge ve Smoothing ---")
    flow_path, output_path, plot_path = get_file_paths()

    if not os.path.exists(flow_path):
        print("❌ HATA: optical_flow.npy bulunamadı. Stage 3'ü çalıştır.")
        return

    # 1. Flow Verisini Yükle
    # Shape: (Frames, H, W, 2)
    flow = np.load(flow_path)
    print(f"📂 Flow Verisi Yüklendi: {flow.shape}")

    num_frames = flow.shape[0]

    # 2. Global Motion Tahmini (Basit ve Etkili Yöntem)
    # Her karedeki tüm flow vektörlerinin MEDYAN'ını alıyoruz.
    # Neden Ortalama (Mean) değil? Çünkü sahneden geçen büyük bir nesne (araba, insan)
    # ortalamayı bozar. Medyan, arka plan hareketini (kamera hareketini) daha iyi temsil eder.

    global_shifts = []
    print("🚀 Global hareket hesaplanıyor (Median Flow)...")

    for i in range(num_frames):
        # O anki karenin dx ve dy katmanları
        dx_map = flow[i, ..., 0]
        dy_map = flow[i, ..., 1]

        # Medyan al (Kameranın ne kadar kaydığını bul)
        dx_global = np.median(dx_map)
        dy_global = np.median(dy_map)

        global_shifts.append([dx_global, dy_global])

    global_shifts = np.array(global_shifts)  # (N, 2)

    # 3. Yörünge (Trajectory) Oluşturma
    # dx, dy'leri toplayarak (cumulative sum) kümülatif yolu buluyoruz.
    # (x_t, y_t) koordinatları
    trajectory = np.cumsum(global_shifts, axis=0)

    # 4. Smoothing (Sanal Kamera Yolu)
    # SensorFlow'daki gibi "Pre-stabilization" mantığı.
    # Sigma değeri ne kadar artarsa yol o kadar "yumuşak" olur.
    SIGMA = 15
    smooth_trajectory = gaussian_filter1d(trajectory, sigma=SIGMA, axis=0)

    # 5. Görsel Jitter (Ground Truth Jitter)
    # Kameranın gerçekte yaptığı hareket - Olması gereken pürüzsüz hareket
    visual_jitter = trajectory - smooth_trajectory

    # Verileri paketleyip kaydet
    # [dx, dy, traj_x, traj_y, smooth_x, smooth_y, jitter_x, jitter_y]
    final_data = np.hstack([
        global_shifts,  # 0, 1
        trajectory,  # 2, 3
        smooth_trajectory,  # 4, 5
        visual_jitter  # 6, 7
    ])

    np.save(output_path, final_data)
    print(f"✅ Görsel yörünge verisi kaydedildi: {output_path}")
    print(f"   Shape: {final_data.shape} (N_frames x 8)")

    # --- Görselleştirme ---
    plt.figure(figsize=(12, 10))

    # X Ekseni Hareketi
    plt.subplot(2, 1, 1)
    plt.title(f"Kamera Yörüngesi (X Ekseni) - Sigma={SIGMA}")
    plt.plot(trajectory[:, 0], label='Orijinal (Titrek)', alpha=0.6, color='gray')
    plt.plot(smooth_trajectory[:, 0], label='Smooth (Hedef)', color='green', linewidth=2)
    plt.legend()
    plt.grid(True)

    # Çıkarılan Jitter
    plt.subplot(2, 1, 2)
    plt.title("Görüntüden Elde Edilen Jitter (Görsel Sarsıntı)")
    plt.plot(visual_jitter[:, 0], label='Visual Jitter (X)', color='red')
    plt.plot(visual_jitter[:, 1], label='Visual Jitter (Y)', color='blue', alpha=0.5)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 Analiz grafiği oluşturuldu: {plot_path}")


if __name__ == "__main__":
    compute_trajectory()