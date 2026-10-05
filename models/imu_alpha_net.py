import torch
import torch.nn as nn


class IMUAdaptiveStabilizerNet(nn.Module):
    """
    IMU tabanlı çoklu görevli 1D CNN.

    Girdi:
        (Batch, Time, 3)
        -> jitter_x, jitter_y, jitter_z

    Çıktı:
        motion:
            (Batch, Time, 2)
            -> dx, dy

        alpha:
            (Batch, Time, 1)
            -> adaptive stabilization intensity
    """

    def __init__(self, dilations=(1, 1, 1, 1), dropout=0.0, in_channels=3, motion_dim=2):
        """
        dilations: conv1, conv2, conv3 ve çıkış başlıkları için genişleme (dilation) katsayıları.
            (1, 1, 1, 1) -> orijinal model (alıcı alan 17 kare ≈ 0.6 sn).
            (1, 2, 4, 8) -> aynı ağırlık sayısı, alıcı alan 61 kare ≈ 2 sn (IDA'daki ~0.7 Hz yalpa için gerekli).
        Ağırlık boyutları dilation'dan bağımsızdır; eski checkpoint'ler varsayılanla aynen yüklenir.
        in_channels: IMU girdi kanalı (3 = jitter x,y,z; 5 = + roll, pitch eğim açıları)
        motion_dim : hareket çıktısı (2 = dx, dy; 3 = dx, dy, dθ)
        """
        super().__init__()
        d1, d2, d3, d4 = dilations
        self.dilations = tuple(dilations)

        # Feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=5,
            padding=2 * d1,
            dilation=d1
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2 * d2,
            dilation=d2
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            padding=2 * d3,
            dilation=d3
        )

        self.relu = nn.ReLU()
        # Dropout parametresizdir; eski checkpoint'lerle uyumluluğu bozmaz (varsayılan 0 = kapalı)
        self.drop = nn.Dropout(dropout)

        # -------------------------------------------------
        # Task 1: Motion correction
        # -------------------------------------------------

        self.motion_head = nn.Conv1d(
            in_channels=32,
            out_channels=motion_dim,
            kernel_size=5,
            padding=2 * d4,
            dilation=d4
        )

        # -------------------------------------------------
        # Task 2: Adaptive alpha
        # -------------------------------------------------

        self.alpha_head = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=5,
            padding=2 * d4,
            dilation=d4
        )

    def forward(self, x):
        """
        Args:
            x:
                (Batch, Time, 3)

        Returns:
            motion:
                (Batch, Time, 2)

            alpha:
                (Batch, Time, 1)
        """

        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        # Shared feature extractor
        x = self.relu(
            self.bn1(self.conv1(x))
        )

        x = self.drop(self.relu(
            self.bn2(self.conv2(x))
        ))

        x = self.drop(self.relu(
            self.conv3(x)
        ))

        # Two task-specific heads
        motion = self.motion_head(x)

        # Sigmoid guarantees:
        # 0 <= alpha <= 1
        alpha = torch.sigmoid(
            self.alpha_head(x)
        )

        # [B,C,T] -> [B,T,C]
        motion = motion.permute(0, 2, 1)
        alpha = alpha.permute(0, 2, 1)

        return motion, alpha


if __name__ == "__main__":

    model = IMUAdaptiveStabilizerNet()

    dummy_input = torch.randn(
        5, 60, 3
    )

    motion, alpha = model(dummy_input)

    print(
        "Motion output:",
        motion.shape
    )

    print(
        "Alpha output:",
        alpha.shape
    )