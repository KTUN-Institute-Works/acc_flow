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

    def __init__(self):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU()

        # -------------------------------------------------
        # Task 1: Motion correction
        # -------------------------------------------------

        self.motion_head = nn.Conv1d(
            in_channels=32,
            out_channels=2,
            kernel_size=5,
            padding=2
        )

        # -------------------------------------------------
        # Task 2: Adaptive alpha
        # -------------------------------------------------

        self.alpha_head = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=5,
            padding=2
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

        x = self.relu(
            self.bn2(self.conv2(x))
        )

        x = self.relu(
            self.conv3(x)
        )

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