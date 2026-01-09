import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def get_file_paths():
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    input_path = os.path.join(project_root, "data", "sensors", "synced_sensors.csv")
    output_path = os.path.join(project_root, "data", "sensors", "processed_features.csv")
    plot_path = os.path.join(project_root, "outputs", "imu_analysis.png")

    return input_path, output_path, plot_path


def apply_high_pass_filter(data, cutoff, fs, order=4):
    """
    Yüksek geçiren filtre (High-Pass): Yerçekimini ve yavaş hareketleri siler,
    geriye sadece ani sarsıntılar (Jitter) kalır.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # filtfilt: Faz kaymasını (gecikmeyi) önlemek için veriyi ileri-geri filtreler
    y = filtfilt(b, a, data)
    return y


def apply_low_pass_filter(data, cutoff, fs, order=4):
    """
    Alçak geçiren filtre (Low-Pass): Titreşimleri siler, ana hareketi ve yerçekimini bırakır.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_imu():
    input_path, output_path, plot_path = get_file_paths()
    print(f"--- STAGE 2: IMU Özellik Çıkarımı (Smoothing & Jitter) ---")

    if not os.path.exists(input_path):
        print("❌ HATA: Synced sensors dosyası bulunamadı. Önce Stage 1'i çalıştır.")
        return

    df = pd.read_csv(input_path)

    # Örnekleme frekansını (FPS) hesapla
    # Stage 1'de video zamanına eşitlemiştik, yani frekans = Video FPS
    timestamps = df['timestamp_sec'].values
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt
    print(f"📡 Sinyal Frekansı (Video FPS): {fs:.2f} Hz")

    # Filtre Ayarları
    # Jitter için genelde 1Hz - 2Hz altını kesip atmak gerekir (Yerçekimi DC'dir, yani 0Hz).
    CUTOFF_FREQ = 2.0  # 2 Hz'in üzerindeki her şey "Titreşim" kabul edilecek.

    features = pd.DataFrame()
    features['frame_idx'] = df['frame_idx']
    features['timestamp_sec'] = df['timestamp_sec']

    # X, Y, Z eksenleri için işlem yap
    for axis in ['x', 'y', 'z']:
        raw_col = f'acc_{axis}'
        raw_data = df[raw_col].values

        # 1. Low Pass (Smooth Motion / Gravity)
        smooth_data = apply_low_pass_filter(raw_data, cutoff=CUTOFF_FREQ, fs=fs)

        # 2. High Pass (Jitter / Noise)
        jitter_data = apply_high_pass_filter(raw_data, cutoff=CUTOFF_FREQ, fs=fs)

        # Dataframe'e ekle
        features[f'smooth_{axis}'] = smooth_data
        features[f'jitter_{axis}'] = jitter_data

    # Ekstra: Toplam Jitter Enerjisi (Magnitude)
    # CNN'in "ne kadar sarsıntı var?" sorusuna tek bir sayıyla cevap verebilmesi için.
    features['jitter_magnitude'] = np.sqrt(
        features['jitter_x'] ** 2 + features['jitter_y'] ** 2 + features['jitter_z'] ** 2
    )

    # Kaydet
    features.to_csv(output_path, index=False)
    print(f"✅ İşlenmiş özellikler kaydedildi: {output_path}")
    print(f"   (İçerik: smooth_x/y/z ve jitter_x/y/z)")

    # --- Görselleştirme (Analiz için çok önemli) ---
    plt.figure(figsize=(12, 6))

    # Sadece X eksenini çizelim (Karmaşa olmasın)
    plt.subplot(2, 1, 1)
    plt.title("X Ekseni: Ham Veri vs. Smooth (Yerçekimi+Hareket)")
    plt.plot(df['timestamp_sec'], df['acc_x'], label='Ham (Raw)', alpha=0.5, color='gray')
    plt.plot(df['timestamp_sec'], features['smooth_x'], label='Smooth (Low-Pass)', color='blue', linewidth=2)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title(f"X Ekseni: Ayrıştırılmış Jitter (High-Pass > {CUTOFF_FREQ}Hz)")
    plt.plot(df['timestamp_sec'], features['jitter_x'], label='Jitter (Noise)', color='red')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 Analiz grafiği oluşturuldu: {plot_path}")
    # plt.show() # Sunucuda çalışmıyorsa kapalı kalsın


if __name__ == "__main__":
    process_imu()