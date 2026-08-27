import torch
import torch.nn as nn


class IMUStabilizerNet(nn.Module):
    """
        IMU (İvmeölçer/Jiroskop) sensör verilerini (zaman serisi) kullanarak,
        videodaki x ve y eksenlerindeki görsel piksel kaymalarını tahmin eden
        1 Boyutlu Evrişimli Sinir Ağı (1D CNN) modeli.

        Mimari Özellikleri:
        - 4 adet 1D Evrişim (Conv1D) katmanından oluşur.
        - Kernel boyutu (5) ve Padding (2) sayesinde Sequence-to-Sequence (Girdi uzunluğu = Çıktı uzunluğu)
          yapısını korur.
        - Ağ, 3 kanallı (x, y, z sarsıntı) veriyi önce 64 kanala kadar genişleterek
          gizli özellikleri öğrenir, ardından 2 kanala (x, y görsel kayma) sıkıştırır.
        - Regresyon (sayısal tahmin) problemi olduğu için çıktı katmanında aktivasyon
          fonksiyonu (Softmax/ReLU vb.) kullanılmaz.
        """
    def __init__(self):
        super(IMUStabilizerNet, self).__init__()

        # --- MİMARİ TASARIMI ---
        # Girdi: (Batch, 3, Time) -> 3 Kanal: acc_x, acc_y, acc_z
        # Çıktı: (Batch, 2, Time) -> 2 Kanal: shift_x, shift_y

        # Katman 1: Özellik Çıkarımı
        # 3 kanaldan 32 kanala genişletiyoruz.
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)  # Stabil eğitim için
        self.relu = nn.ReLU()

        # Katman 2: Derinlik (Non-linearity)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Katman 3: Darboğaz (Bottleneck)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding=2)

        # Katman 4: Çıktı Katmanı (Regression)
        # Sonuçta X ve Y düzeltmesi istiyoruz (2 kanal)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=2, kernel_size=5, padding=2)

    def forward(self, x):
        """
        Modelin ileri yayılım (forward pass) işlemi.

        Args:
            x (torch.Tensor): Modele giren IMU verisi.
                              Beklenen boyut: (Batch_Size, Sequence_Length, 3)

        Returns:
            torch.Tensor: Tahmin edilen görsel kayma (shift) miktarı.
                          Çıktı boyutu: (Batch_Size, Sequence_Length, 2)
        """
        # PyTorch Conv1D [Batch, Channels, Length] bekler.
        # Bizim verimiz [Batch, Length, Channels] (19, 60, 3).
        # Bu yüzden önce boyutları değiştirmeliyiz (Permute).

        # (Batch, Length, Channels) -> (Batch, Channels, Length)
        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))

        # Son katmanda aktivasyon yok (Linear output)
        x = self.conv4(x)

        # Çıktıyı eski formatına döndür: (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        return x


if __name__ == "__main__":
    # Test (Dummy Input)
    model = IMUStabilizerNet()
    dummy_input = torch.randn(5, 60, 3)  # 5 örnek, 60 zaman, 3 özellik
    output = model(dummy_input)
    print(f"Model Test Output Shape: {output.shape}")
    # Beklenen: (5, 60, 2)