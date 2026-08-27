import pandas as pd
import numpy as np
import os


def get_file_paths():
    """
    Stage 6 (Veri Seti Hazırlığı) için dosya yollarını üretir.

    Döndürdüğü yollar:
    - imu_features_path: Yapay zekaya GİRDİ (X) olacak IMU özellik dosyası (Stage 2'den).
    - trajectory_path: Yapay zekaya HEDEF (Y) olacak görsel sarsıntı dosyası (Stage 4'ten).
    - output_x_path: Pencereleme (windowing) işleminden geçmiş X veri setinin .npy yolu.
    - output_y_path: Pencereleme işleminden geçmiş Y etiket setinin .npy yolu.
    """
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    imu_features_path = os.path.join(project_root, "data", "sensors", "processed_features.csv")
    trajectory_path = os.path.join(project_root, "data", "videos", "visual_trajectory.npy")

    output_x_path = os.path.join(project_root, "data", "dataset_X.npy")
    output_y_path = os.path.join(project_root, "data", "dataset_Y.npy")

    return imu_features_path, trajectory_path, output_x_path, output_y_path


def build_windows():
    """
    Zaman serisi olan IMU ve Optik Akış verilerini, Derin Öğrenme (CNN/LSTM)
    modellerinin eğitilebilmesi için "Sliding Window" (Kayan Pencere) yöntemiyle
    3 boyutlu tensör (matris) bloklarına dönüştürür.

    İşlem Adımları:
    1. IMU Jitter (X) ve Visual Jitter (Y) verileri yüklenir.
    2. İki veri seti arasındaki olası 1-2 karelik farklar (min_len) bulunup uçtan kırpılarak
       birebir eşit (1:1) eşleşme garanti altına alınır.
    3. WINDOW_SIZE (ör. 60 kare) genişliğinde pencereler oluşturulur.
    4. Pencereler STRIDE (ör. 5 kare) adımıyla kaydırılarak veri çoğaltılır (overlap).
    5. Elde edilen X (Input) ve Y (Target) dizileri Numpy array formatında (.npy)
       modellemeye hazır halde kaydedilir.
    """
    print(f"--- STAGE 6: Veri Seti Hazırlığı (Windowing) ---")
    imu_path, traj_path, out_x, out_y = get_file_paths()

    # 1. Verileri Yükle
    if not os.path.exists(imu_path) or not os.path.exists(traj_path):
        print("HATA: Girdi dosyaları eksik (Stage 2 veya Stage 4 çalışmamış).")
        return

    # IMU Verisi (Stage 2'den gelen jitter_x, jitter_y, jitter_z)
    df_imu = pd.read_csv(imu_path)

    # Görsel Yörünge Verisi (Stage 4'ten gelen)
    # [dx, dy, traj_x, traj_y, smooth_x, smooth_y, jitter_x, jitter_y]
    # Bize son iki sütun lazım: Visual Jitter (X, Y)
    visual_data = np.load(traj_path)
    visual_jitter = visual_data[:, 6:8]  # Sütun 6 ve 7

    # 2. Boyut Kontrolü (Senkronizasyon Teyidi)
    n_imu = len(df_imu)
    n_vis = len(visual_jitter)
    print(f"Veri Uzunlukları -> IMU: {n_imu}, Visual: {n_vis}")

    # En kısa olana göre kırp (Eğer 1-2 frame fark varsa sondan atalım)
    min_len = min(n_imu, n_vis)
    df_imu = df_imu.iloc[:min_len]
    visual_jitter = visual_jitter[:min_len]

    # 3. Girdi ve Çıktı Dizilerini Oluştur
    # X (Input): IMU Jitter (High Frequency Accelerometer)
    # Sadece jitter sütunlarını alıyoruz.
    # (Opsiyonel: smooth verisini de ekleyebiliriz ama şimdilik jitter odaklı gidelim)
    X_full = df_imu[['jitter_x', 'jitter_y', 'jitter_z']].values

    # Y (Target): Visual Jitter (Görüntüden hesaplanan düzeltme)
    # Method A doğrulandığı için direkt visual_jitter'ı hedef olarak alıyoruz.
    Y_full = visual_jitter

    # 4. Pencereleme (Sliding Window)
    # CNN'in geçmişe ve geleceğe bakabilmesi için bir pencere genişliği seçiyoruz.
    WINDOW_SIZE = 60  # 60 kare = ~2 saniye (30 FPS'de)
    STRIDE = 5  # 5 karede bir yeni pencere al (Veri çoğaltma / Augmentation)

    windows_X = []
    windows_Y = []

    print(f"Pencereleme Başlıyor (Size={WINDOW_SIZE}, Stride={STRIDE})...")

    for i in range(0, min_len - WINDOW_SIZE, STRIDE):
        # Pencereyi kes
        x_window = X_full[i: i + WINDOW_SIZE]
        y_window = Y_full[i: i + WINDOW_SIZE]

        windows_X.append(x_window)
        windows_Y.append(y_window)

    # Numpy array'e çevir
    X_dataset = np.array(windows_X, dtype=np.float32)
    Y_dataset = np.array(windows_Y, dtype=np.float32)

    # 5. Kaydet
    np.save(out_x, X_dataset)
    np.save(out_y, Y_dataset)

    print(f"\nVeri seti oluşturuldu!")
    print(f"   X (Input) Shape: {X_dataset.shape} -> (Örnek, Süre, Özellik=3)")
    print(f"   Y (Label) Shape: {Y_dataset.shape} -> (Örnek, Süre, Çıktı=2)")
    print(f"   Kaydedilen yer: data/dataset_X.npy")


if __name__ == "__main__":
    build_windows()