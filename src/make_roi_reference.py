import cv2
from pathlib import Path

video_path = "data/entrance.mp4"
output_path = "outputs/inspection/entrance_roi_reference.jpg"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

# Use a frame around the middle of the video.
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = total_frames // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read reference frame.")

# Draw coordinate grid.
for x in range(0, frame.shape[1], 100):
    cv2.line(frame, (x, 0), (x, frame.shape[0]), (180, 180, 180), 1)
    cv2.putText(
        frame,
        str(x),
        (x + 3, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

for y in range(0, frame.shape[0], 100):
    cv2.line(frame, (0, y), (frame.shape[1], y), (180, 180, 180), 1)
    cv2.putText(
        frame,
        str(y),
        (5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(output_path, frame)

print(f"Saved: {output_path}")
print(f"Reference frame: {frame_number}/{total_frames}")