import cv2
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


def get_file_paths():
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    imu_path = os.path.join(project_root, "data", "sensors", "sensors.csv")
    output_path = os.path.join(project_root, "data", "sensors", "synced_sensors.csv")

    return video_path, imu_path, output_path


def sync_data():
    """
        Video ve IMU sensör verilerini zaman ekseninde birbirine hizalar (senkronize eder).

        İşlem Adımları:
        1. Videonun FPS ve frame sayısını kullanarak her bir karenin (frame) ideal
           saniyesini (0.0, 0.033, 0.066...) hesaplar.
        2. IMU verisindeki tekrar eden (aynı milisaniyeye sahip) zaman damgalarını,
           değerlerin ortalamasını (mean) alarak tekilleştirir ve temizler.
        3. Sensör zamanını saniyeye çevirip başlangıç noktasını (t=0) videoyla eşler.
        4. Doğrusal interpolasyon (linear interpolation) kullanarak, sensör verilerini
           tam olarak video karelerinin olduğu zaman damgalarına göre yeniden örnekler (resampling).
        5. Sonuçları, her bir video karesi için bir satır olacak şekilde
           'synced_sensors.csv' dosyasına kaydeder.
    """
    video_path, imu_path, output_path = get_file_paths()
    print(f"--- STAGE 1: IMU & Video Senkronizasyonu ---")


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_timestamps = np.arange(0, frame_count) / fps
    print(f" Video: {frame_count} frames @ {fps} FPS")

    df = pd.read_csv(imu_path)
    df.columns = [c.strip() for c in df.columns]  # Boşluk temizliği

    df_clean = df.groupby('timestamp_ms', as_index=False).mean()

    print(f" IMU Ham Veri: {len(df)} satır")
    print(f" Temizlenmiş (Dedup) Veri: {len(df_clean)} satır")

    t_imu = df_clean['timestamp_ms'].values / 1000.0
    t_imu = t_imu - t_imu[0]  # Başlangıcı 0 yap

    imu_vals = df_clean[['x', 'y', 'z']].values

    print(" İnterpolasyon yapılıyor (Video karelerine hizalanıyor)...")

    # Fonksiyonları oluştur
    interp_func_x = interp1d(t_imu, imu_vals[:, 0], kind='linear', fill_value="extrapolate")
    interp_func_y = interp1d(t_imu, imu_vals[:, 1], kind='linear', fill_value="extrapolate")
    interp_func_z = interp1d(t_imu, imu_vals[:, 2], kind='linear', fill_value="extrapolate")

    # Video zamanlarında sensör değerlerini hesapla
    synced_x = interp_func_x(video_timestamps)
    synced_y = interp_func_y(video_timestamps)
    synced_z = interp_func_z(video_timestamps)

    # 4. Kaydet
    df_synced = pd.DataFrame({
        'frame_idx': range(frame_count),
        'timestamp_sec': video_timestamps,
        'acc_x': synced_x,
        'acc_y': synced_y,
        'acc_z': synced_z
    })

    df_synced.to_csv(output_path, index=False)
    print(f" Senkronize veri kaydedildi: {output_path}")
    print(df_synced.head())


if __name__ == "__main__":
    sync_data()