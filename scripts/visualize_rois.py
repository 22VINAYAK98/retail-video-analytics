import argparse
from pathlib import Path

import cv2
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_roi(frame, name, polygon, label_position=None):
    points = [(int(x), int(y)) for x, y in polygon]

    # Draw polygon
    cv2.polylines(
        frame,
        [__import__("numpy").array(points, dtype="int32")],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    # Label position
    if label_position is None:
        label_position = points[0]

    cv2.putText(
        frame,
        name,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw vertices
    for i, (x, y) in enumerate(points):
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            frame,
            str(i),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ROIs from entrance.yaml"
    )

    parser.add_argument(
        "--config",
        default="configs/entrance.yaml",
        help="Path to ROI YAML configuration",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=8540,
        help="Frame number to visualize",
    )

    parser.add_argument(
        "--output",
        default="outputs/inspection/entrance_rois.jpg",
        help="Output image path",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config = load_config(args.config)

    video_path = config["video"]["path"]
    rois = config["rois"]

    print(f"Config : {args.config}")
    print(f"Video  : {video_path}")
    print(f"Frame  : {args.frame}")

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"Video resolution: {width} x {height}")
    print(f"Total frames    : {total_frames}")

    if args.frame < 0 or args.frame >= total_frames:
        raise ValueError(
            f"Frame {args.frame} is outside video "
            f"range 0-{total_frames - 1}"
        )

    # ---------------------------------------------------------
    # Read selected frame
    # ---------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        args.frame
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {args.frame}"
        )

    # ---------------------------------------------------------
    # Draw ROIs
    # ---------------------------------------------------------

    for name, roi_config in rois.items():

        polygon = roi_config["polygon"]

        draw_roi(
            frame,
            name,
            polygon,
        )

        print(f"\n{name}:")
        for point in polygon:
            print(f"  {point}")

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        frame,
    )

    if not success:
        raise RuntimeError(
            f"Could not save output: {output_path}"
        )

    print(
        f"\nROI visualization saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()