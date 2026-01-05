import cv2
import os

video_path = "../videos/ms/wtf25_kv_syq_semifinals/wtf25_kv_syq_semifinals-test-seg1.mp4"
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (512, 288))
    cv2.imwrite(f"{out_dir}/{i:06d}.jpg", frame)
    i += 1

cap.release()