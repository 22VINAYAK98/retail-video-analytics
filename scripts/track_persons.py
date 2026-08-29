import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/entrance.mp4"
OUTPUT_PATH = "outputs/person_tracking.mp4"

MODEL_PATH = "yolo26m.pt"  # "yolo11n.pt"

CONFIDENCE = 0.25
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

        results = model.track(
            frame,
            persist=True,
            classes=[0],          # person
            conf=CONFIDENCE,
            tracker= "botsort.yaml", # "confyolo26m.ptigs/botsort.yaml",  # "configs/bytetrack.yaml",  #"bytetrack.yaml",
            verbose=False,
            device=0,
        )

        annotated = frame.copy()

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().numpy()

            for box, track_id, confidence in zip(
                xyxy, track_ids, confidences
            ):
                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                # Bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # Track ID
                label = f"ID {track_id} {confidence:.2f}"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Bottom-center point
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