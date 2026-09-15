import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import sys


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(
    os.path.dirname(current_script_path)
)

sys.path.append(project_root)

from models.imu_alpha_net import (
    IMUAdaptiveStabilizerNet
)


def get_file_paths():

    x_path = os.path.join(
        project_root,
        "data",
        "dataset_X.npy"
    )

    y_path = os.path.join(
        project_root,
        "data",
        "dataset_Y.npy"
    )

    model_save_path = os.path.join(
        project_root,
        "models",
        "best_imu_adaptive_model.pth"
    )

    loss_plot_path = os.path.join(
        project_root,
        "outputs",
        "training_loss_adaptive.png"
    )

    return (
        x_path,
        y_path,
        model_save_path,
        loss_plot_path
    )


def train_model():

    print(
        "--- STAGE 7: "
        "Multi-Task IMU CNN Eğitimi ---"
    )

    (
        x_path,
        y_path,
        model_save_path,
        loss_plot_path
    ) = get_file_paths()

    if not os.path.exists(x_path):
        print(
            "HATA: dataset_X.npy bulunamadı."
        )
        return

    if not os.path.exists(y_path):
        print(
            "HATA: dataset_Y.npy bulunamadı."
        )
        return

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    X_numpy = np.load(x_path)
    Y_numpy = np.load(y_path)

    print(
        "X shape:",
        X_numpy.shape
    )

    print(
        "Y shape:",
        Y_numpy.shape
    )

    X_tensor = torch.from_numpy(
        X_numpy
    ).float()

    Y_tensor = torch.from_numpy(
        Y_numpy
    ).float()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Eğitim cihazı:",
        device
    )

    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = IMUAdaptiveStabilizerNet().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -------------------------------------------------
    # Loss weights
    # -------------------------------------------------

    LAMBDA_MOTION = 1.0
    LAMBDA_ALPHA = 0.5

    motion_criterion = nn.MSELoss()

    alpha_criterion = nn.MSELoss()

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    EPOCHS = 500

    total_loss_history = []
    motion_loss_history = []
    alpha_loss_history = []

    best_loss = float("inf")

    print(
        f"Eğitim başladı ({EPOCHS} epoch)..."
    )

    model.train()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        # Forward
        pred_motion, pred_alpha = model(
            X_tensor
        )

        # Ground truth
        target_motion = Y_tensor[:, :, :2]

        target_alpha = Y_tensor[:, :, 2:3]

        # Motion loss
        motion_loss = motion_criterion(
            pred_motion,
            target_motion
        )

        # Alpha loss
        alpha_loss = alpha_criterion(
            pred_alpha,
            target_alpha
        )

        # Multi-task loss
        loss = (
            LAMBDA_MOTION * motion_loss
            +
            LAMBDA_ALPHA * alpha_loss
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        total_loss_history.append(
            loss.item()
        )

        motion_loss_history.append(
            motion_loss.item()
        )

        alpha_loss_history.append(
            alpha_loss.item()
        )

        # Best model
        if loss.item() < best_loss:

            best_loss = loss.item()

            torch.save(
                model.state_dict(),
                model_save_path
            )

        if (epoch + 1) % 50 == 0:

            print(
                f"Epoch "
                f"[{epoch + 1}/{EPOCHS}] | "
                f"Total: {loss.item():.6f} | "
                f"Motion: {motion_loss.item():.6f} | "
                f"Alpha: {alpha_loss.item():.6f}"
            )

    print(
        "\nModel kaydedildi:"
    )

    print(model_save_path)

    # -------------------------------------------------
    # Loss plot
    # -------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        total_loss_history,
        label="Total Loss"
    )

    plt.plot(
        motion_loss_history,
        label="Motion Loss"
    )

    plt.plot(
        alpha_loss_history,
        label="Alpha Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Multi-Task CNN Training Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        loss_plot_path
    )

    plt.close()

    print(
        "Loss grafiği:",
        loss_plot_path
    )

    # -------------------------------------------------
    # Example prediction
    # -------------------------------------------------

    model.eval()

    with torch.no_grad():

        sample_x = X_tensor[0:1]

        pred_motion, pred_alpha = model(
            sample_x
        )

        target_motion = (
            Y_tensor[0:1, :, :2]
        )

        target_alpha = (
            Y_tensor[0:1, :, 2:3]
        )

    print(
        "\nİlk pencere:"
    )

    print(
        "Motion target mean:",
        target_motion.mean().item()
    )

    print(
        "Motion prediction mean:",
        pred_motion.mean().item()
    )

    print(
        "Alpha target mean:",
        target_alpha.mean().item()
    )

    print(
        "Alpha prediction mean:",
        pred_alpha.mean().item()
    )


if __name__ == "__main__":
    train_model()