<h1 align="left">ACCFlow: Deep Sensor-Fusion Video Stabilization <a href="#"><img src="https://img.shields.io/badge/Status-Completed-<COLOR>.svg" ></a> </h1> 

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#results-demo">Results Demo</a> |
  <a href="#installation">Installation</a> |
  <a href="#pipeline-usage">Pipeline & Usage</a> |
  <a href="#news">News</a> |
  <a href="#statement">Statement</a> |
  <a href="#reference">Reference</a>
</p>

## Introduction

<p align="justify">This repository contains the code, models, and data processing pipeline for <b>IMU-Alpha-Net</b>, a robust video stabilization framework. Unlike traditional methods that rely solely on heavy optical flow computations at runtime, this project utilizes a Sensor-Fusion approach. We use high-accuracy Optical Flow (RAFT) and IMU (Accelerometer/Gyroscope) data during the <b>training phase</b> to teach a 1D Convolutional Neural Network (CNN) how physical device jitter translates to visual pixel drift.</p>

<p align="justify">Once trained, the <code>IMUStabilizerNet</code> model can stabilize shaky videos ultra-fast by <b>only looking at the IMU data</b> during inference, bypassing the need for heavy visual processing. The pipeline handles everything from temporal synchronization and Butterworth filtering to dynamic trajectory smoothing and adaptive cropping.</p>

<img src="outputs/final_comparison_plot.png" alt="CNN Prediction vs Ground Truth">

## Results Demo

We test IMU-Alpha-Net on shaky handheld videos. Below is an overview of our adaptive stabilization and inference results (Original vs. Stabilized vs. Overlay).

<img src="demo/3panel_demo.gif" width='100%' alt="3 Panel Comparison">

For the ground truth generation, we utilize **RAFT (Recurrent All-Pairs Field Transforms)** to compute dense optical flow, taking the median motion to estimate global camera trajectories robustly, ignoring foreground moving objects.

<img src="demo/raft_flow.gif" class="left" width='45%'><img src="demo/trajectory_smoothing.gif" class="right" width='45%'>

During inference, our 1D CNN predicts the dynamic visual jitter directly from the high-pass filtered IMU data, applying an inverse warp and global crop to yield a perfectly stable output without black borders.

## Installation
Requirements:

- Python 3.8+
- Pytorch (version 1.8.0+)
- Torchvision
- OpenCV (cv2)
- Pandas, Numpy, Scipy, Matplotlib

1. Clone this repository and the RAFT submodule

    `git clone https://github.com/KTUN-Institute-Works/acc_flow`

2. Go into the repository

    `cd IMU-Alpha-Net`

3. Create conda environment and activate

    `conda create -n imu_stab python=3.8`
    
    `conda activate imu_stab`

4. Install dependencies
    
    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    
    `pip install -r requirements.txt`


## Pipeline & Usage

Our framework is strictly modularized into 8 stages. Place your shaky video at `data/videos/input.mp4` and sensor data at `data/sensors/sensors.csv`.

**Data Preparation & Ground Truth Extraction:**
* `python stage0_inspect_data.py` - Verifies integrity and FPS/Hz of the input data.
* `python stage1_sync_imu_data.py` - Interpolates and temporal-syncs IMU to video frames.
* `python stage2_imu_smoothing.py` - Applies Butterworth Low/High-pass filters to extract physical jitter.
* `python stage3_compute_flow.py` - Runs RAFT to extract dense optical flow.
* `python stage4_smooth_flow.py` - Computes Gaussian-smoothed global visual trajectories and visual jitter (Ground Truth).
* `python stage5_warp_video.py` - Debug script to visualize stabilization mathematical hypothesis via 2x2 grid.

**Training:**
* `python stage6_build_windows.py` - Builds overlapping sliding windows (X: IMU Jitter, Y: Visual Jitter) for the CNN.
* `python stage7_train_imu_cnn.py` - Trains the 1D Fully Convolutional Network (`IMUStabilizerNet`) and saves the weights.

**Inference (Testing):**
* `python stage8_blend.py` - Runs inference on the full video using the trained model, applying adaptive warping and global cropping. Results are saved in the `outputs/` folder.

## News
- [x] Release data synchronization and temporal alignment code (Stage 0-1).
- [x] Integrate RAFT for robust visual trajectory extraction (Stage 3-4).
- [x] Build and train `IMUStabilizerNet` 1D CNN (Stage 6-7).
- [x] Release ultra-fast IMU-only inference and adaptive cropping (Stage 8).
- [ ] Add support for Gyroscope-based 3D rotation estimations.
- [ ] Release pre-trained weights for general mobile devices.

## Statement
This project is developed for research and educational purposes in the fields of Sensor Fusion and Computer Vision. For questions, suggestions, or commercial inquiries, please contact [murabitakdogan@gmail.com](mailto:murabitakdogan@gmail.com).

## Reference
- <a href="https://github.com/princeton-vl/RAFT">RAFT: Recurrent All-Pairs Field Transforms for Optical Flow</a>
- <a href="https://pytorch.org/">PyTorch</a>
