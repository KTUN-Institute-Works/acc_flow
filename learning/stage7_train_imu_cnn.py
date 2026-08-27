import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Model sınıfını import et
# (Proje kök dizinini path'e eklememiz gerekebilir)
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

from models.imu_alpha_net import IMUStabilizerNet


def get_file_paths():
    """
    Stage 7 (Eğitim) aşaması için gerekli giriş/çıkış dosya yollarını üretir.

    Döndürdüğü yollar:
    - x_path: Modelin girdisi olan IMU özellikleri (Stage 6'dan).
    - y_path: Modelin hedefi/etiketi olan Visual Jitter (Stage 6'dan).
    - model_save_path: Eğitilen modelin ağırlıklarının (state_dict) kaydedileceği .pth dosyası.
    - loss_plot_path: Eğitim süresince azalan hata (MSE) miktarının grafiği.
    """
    x_path = os.path.join(project_root, "data", "dataset_X.npy")
    y_path = os.path.join(project_root, "data", "dataset_Y.npy")
    model_save_path = os.path.join(project_root, "models", "best_imu_model.pth")
    loss_plot_path = os.path.join(project_root, "outputs", "training_loss.png")
    return x_path, y_path, model_save_path, loss_plot_path


def train_model():
    """
    IMU verilerinden görsel sarsıntıyı tahmin edecek olan Derin Öğrenme (CNN)
    modelini PyTorch kullanarak eğitir.

    İşlem Adımları:
    1. Stage 6'da oluşturulan Numpy veri setleri yüklenir ve PyTorch Tensörlerine çevrilir.
    2. Cihaz (CPU/GPU) kontrolü yapılır ve veriler/model uygun cihaza taşınır.
    3. IMUStabilizerNet modeli, Adam optimizasyon algoritması ve MSE (Ortalama Kare Hatası)
       kayıp fonksiyonu (loss function) başlatılır.
    4. Belirlenen Epoch sayısı kadar eğitim döngüsü çalıştırılır (Forward pass, Loss, Backward pass, Step).
    5. Eğitilen modelin ağırlıkları 'best_imu_model.pth' olarak kaydedilir.
    6. Eğitim sürecinin (Loss değerlerinin) gidişatı bir grafik (PNG) olarak dışa aktarılır.
    7. Başarımı teyit etmek için ilk veri örneği üzerinde hızlı bir tahmin (inference) yapılarak
       hedef ve tahmin ortalamaları konsola yazdırılır.
    """
    print(f"--- STAGE 7: CNN Eğitimi Başlıyor ---")
    x_path, y_path, model_save_path, loss_plot_path = get_file_paths()

    # 1. Veri Yükleme
    if not os.path.exists(x_path):
        print("❌ HATA: Veri seti bulunamadı. Stage 6'yı çalıştır.")
        return

    X_numpy = np.load(x_path)
    Y_numpy = np.load(y_path)

    # Tensor Dönüşümü
    X_tensor = torch.from_numpy(X_numpy).float()
    Y_tensor = torch.from_numpy(Y_numpy).float()

    # Cihaz Seçimi (GPU varsa kullan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Eğitim Cihazı: {device}")

    # 2. Model, Loss, Optimizer
    model = IMUStabilizerNet().to(device)
    criterion = nn.MSELoss()  # Hedefimiz regresyon (sapmayı tahmin etmek)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Veriyi Cihaza Taşı
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # 3. Eğitim Döngüsü
    EPOCHS = 500  # Veri az olduğu için epoch sayısını yüksek tutuyoruz
    loss_history = []

    print(f"🚀 Eğitim başladı ({EPOCHS} Epoch)...")

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(X_tensor)

        # Loss Hesapla
        loss = criterion(outputs, Y_tensor)

        # Backward Pass ve Update
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}")

    # 4. Modeli Kaydet
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Model kaydedildi: {model_save_path}")

    # 5. Loss Grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Eğitim Kaybı (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    print(f"📊 Loss grafiği kaydedildi: {loss_plot_path}")

    # 6. Tahmin Örneği (İlk pencere için)
    model.eval()
    with torch.no_grad():
        sample_in = X_tensor[0:1]  # İlk örnek
        sample_pred = model(sample_in)
        sample_target = Y_tensor[0:1]

        print("\n🔍 Test (İlk Örnek Karşılaştırması):")
        print(f"   Input (IMU Jitter Mean): {sample_in.mean().item():.4f}")
        print(f"   Target (Visual Drift Mean): {sample_target.mean().item():.4f}")
        print(f"   Predicted (Model Mean): {sample_pred.mean().item():.4f}")


if __name__ == "__main__":
    train_model()