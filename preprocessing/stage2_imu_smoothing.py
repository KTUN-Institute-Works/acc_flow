import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def get_file_paths():
    """
    Stage 2 (Filtreleme) işlemi için gerekli giriş, çıkış ve grafik dosyalarının
    yollarını projenin ana dizinine göre dinamik olarak hesaplar.

    Döndürdüğü yollar:\n
    ``input_path:`` Stage 1'den gelen senkronize edilmiş sensör verisi.\n
    ``output_path:`` Çıkarılan yeni özelliklerin (smooth, jitter) kaydedileceği yer.\n
    ``plot_path:`` Filtreleme işleminin görsel analizinin kaydedileceği PNG dosyası.
    """
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
    """
    Senkronize edilmiş IMU verilerine sinyal işleme teknikleri uygulayarak
    anlamlı özellikler (features) çıkarır.

    İşlem Adımları:

    1. Zaman damgalarından sinyalin örnekleme frekansını (fs) hesaplar.

    2. Her bir eksen (X, Y, Z) için Butterworth Low-Pass filtresi uygulayarak yerçekimini ve bilinçli/yavaş hareketleri ('smooth') izole eder.

    3. Her bir eksen için Butterworth High-Pass filtresi uygulayarak ani sarsıntı ve titreşimleri ('jitter') izole eder.

    4. Yapay zeka modelleri için tek bir sarsıntı metriği sağlamak amacıyla X, Y, Z sarsıntılarının bileşkesini (jitter_magnitude) hesaplar.

    5. Tüm bu yeni özellikleri CSV olarak kaydeder ve doğrulama için X eksenindeki değişimi gösteren bir grafik (PNG) çizer.
    """
    input_path, output_path, plot_path = get_file_paths()
    print(f"--- STAGE 2: IMU Özellik Çıkarımı (Smoothing & Jitter) ---")

    if not os.path.exists(input_path):
        print(" HATA: Synced sensors dosyası bulunamadı. Önce Stage 1'i çalıştır.")
        return

    df = pd.read_csv(input_path)

    timestamps = df['timestamp_sec'].values
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt
    print(f" Sinyal Frekansı (Video FPS): {fs:.2f} Hz")

    CUTOFF_FREQ = 2.0

    features = pd.DataFrame()
    features['frame_idx'] = df['frame_idx']
    features['timestamp_sec'] = df['timestamp_sec']

    for axis in ['x', 'y', 'z']:
        raw_col = f'acc_{axis}'
        raw_data = df[raw_col].values

        smooth_data = apply_low_pass_filter(raw_data, cutoff=CUTOFF_FREQ, fs=fs)

        jitter_data = apply_high_pass_filter(raw_data, cutoff=CUTOFF_FREQ, fs=fs)

        features[f'smooth_{axis}'] = smooth_data
        features[f'jitter_{axis}'] = jitter_data

    features['jitter_magnitude'] = np.sqrt(
        features['jitter_x'] ** 2 + features['jitter_y'] ** 2 + features['jitter_z'] ** 2
    )

    features.to_csv(output_path, index=False)
    print(f" İşlenmiş özellikler kaydedildi: {output_path}")
    print(f"   (İçerik: smooth_x/y/z ve jitter_x/y/z)")

    plt.figure(figsize=(12, 6))

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
    print(f" Analiz grafiği oluşturuldu: {plot_path}")


if __name__ == "__main__":
    process_imu()