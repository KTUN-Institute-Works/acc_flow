import cv2
import numpy as np

VIDEO_IN = "../outputs/final_cnn_stabilized.mp4"
VIDEO_OUT = "../outputs/final_cnn_stabilized_cropped.mp4"

cap = cv2.VideoCapture(VIDEO_IN)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

frames = np.stack(frames)
print("Video shape:", frames.shape)  # [N, H, W, 3]

# --- valid mask oluştur ---
# siyah border detect
gray = frames.mean(axis=3)

mask = gray > 5  # threshold

# --- tüm frameler için ortak alan ---
global_mask = np.all(mask, axis=0)

# --- bounding box bul ---
coords = np.argwhere(global_mask)

y1, x1 = coords.min(axis=0)
y2, x2 = coords.max(axis=0)

print("Crop area:", x1, y1, x2, y2)

# küçük margin ekle (güvenlik)
margin = 5
y1 += margin
x1 += margin
y2 -= margin
x2 -= margin

# --- video writer ---
H, W = frames.shape[1:3]
fps = 30  # gerekirse cap'ten al

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    VIDEO_OUT,
    fourcc,
    fps,
    (x2 - x1, y2 - y1)
)

# --- crop uygula ---
for f in frames:
    cropped = f[y1:y2, x1:x2]
    out.write(cropped)

out.release()

print("Cropped video saved:", VIDEO_OUT)