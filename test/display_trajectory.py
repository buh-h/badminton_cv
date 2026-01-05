import cv2
import matplotlib.pyplot as plt

video_path = "video.mp4"
frame_number = 150  # 0-based index

cap = cv2.VideoCapture(video_path)

# Jump directly to the frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame")

# Convert BGR → RGB for display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_rgb)
plt.axis("off")
plt.show()