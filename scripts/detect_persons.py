import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/entrance.mp4"
OUTPUT_PATH = "outputs/person_detection.mp4"

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.4
MAX_FRAMES = 900  # ~30 seconds at 30 FPS


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    frame_count = 0

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            classes=[0],       # COCO class 0 = person
            conf=CONFIDENCE,
            verbose=False,
            device=0,
        )

        annotated = results[0].plot()

        # Draw bottom-center point for each person
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int(y2)

                cv2.circle(
                    annotated,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1,
                )

        writer.write(annotated)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    writer.release()

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_count}")


if __name__ == "__main__":
    main()