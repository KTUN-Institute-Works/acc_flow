import cv2
import pandas as pd
import numpy as np
import os


def get_file_paths():
    """
    Script'in çalıştığı konuma bakmaksızın, dosya yollarını
    proje yapısına göre dinamik olarak hesaplar.
    """
    current_script_path = os.path.abspath(__file__)

    preprocessing_dir = os.path.dirname(current_script_path)

    project_root = os.path.dirname(preprocessing_dir)

    video_path = os.path.join(project_root, "data", "videos", "input.mp4")
    imu_path = os.path.join(project_root, "data", "sensors", "sensors.csv")

    return video_path, imu_path


def inspect_data():
    VIDEO_PATH, IMU_PATH = get_file_paths()

    print(f"--- STAGE 0: Veri Analizi Başlıyor ---")
    print(f"📂 Çalışma Dizini: {os.getcwd()}")
    print(f"📂 Hedef Video Yolu: {VIDEO_PATH}")

    # ---------------------------------------------------------
    # 1. VİDEO ANALİZİ
    # ---------------------------------------------------------
    video_duration = 0
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ HATA: Video dosyası fiziksel olarak yok!")
    elif os.path.getsize(VIDEO_PATH) == 0:
        print(f"❌ HATA: Video dosyası boş (0 byte).")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ HATA: Video dosyası var ama açılamadı (Codec sorunu?).")
        else:
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration = video_frames / video_fps if video_fps > 0 else 0

            print(f"\n🎥 VIDEO BİLGİLERİ:")
            print(f"   Frame Sayısı: {video_frames}")
            print(f"   FPS: {video_fps:.2f}")
            print(f"   Süre: {video_duration:.4f} saniye")
            cap.release()

    # ---------------------------------------------------------
    # 2. IMU SENSÖR ANALİZİ (Özel Format)
    # ---------------------------------------------------------
    if not os.path.exists(IMU_PATH) or os.path.getsize(IMU_PATH) == 0:
        print(f"\n❌ HATA: Sensör dosyası bulunamadı: {IMU_PATH}")
    else:
        try:
            df = pd.read_csv(IMU_PATH)

            df.columns = [c.strip() for c in df.columns]

            print(f"\n📉 IMU SENSÖR BİLGİLERİ:")

            if 'timestamp_ms' in df.columns:
                t_sec = df['timestamp_ms'].values / 1000.0

                imu_duration = t_sec[-1] - t_sec[0]
                num_samples = len(df)

                print(f"   Örnek Sayısı: {num_samples}")
                print(f"   IMU Süresi: {imu_duration:.4f} saniye")

                # Frekans Hesabı
                if imu_duration > 0:
                    avg_freq = num_samples / imu_duration
                    print(f"   Ortalama Frekans: ~{avg_freq:.2f} Hz")

                # Duplicate Kontrolü
                deltas = np.diff(t_sec)
                zero_deltas = np.sum(deltas == 0)

                if zero_deltas > 0:
                    print(f"   ⚠️ UYARI: {zero_deltas} adet örnekte zaman farkı 0 ms.")
                    print(f"      (Stage 1'de resampling yapılacak.)")
                else:
                    print(f"   ✅ Zaman damgaları tekil.")

                # Video ile Kıyaslama
                if video_duration > 0:
                    diff = abs(video_duration - imu_duration)
                    print(f"\n⚖️  SENKRONİZASYON DURUMU:")
                    print(f"   Video: {video_duration:.2f}s | IMU: {imu_duration:.2f}s")
                    print(f"   Fark: {diff:.2f}s")

                print("\n📊 Örnek Veri (İlk 5 Satır):")
                print(df[['timestamp_ms', 'x', 'y', 'z']].head().to_string(index=False))

            else:
                print(f"❌ HATA: 'timestamp_ms' sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")
                print("   CSV dosyanın içeriğini kontrol et.")

        except Exception as e:
            print(f"❌ Beklenmeyen Hata: {e}")


if __name__ == "__main__":
    inspect_data()