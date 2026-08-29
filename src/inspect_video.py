import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, num_frames: int = 12):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise RuntimeError("Video contains no readable frames.")

    frame_indices = [
        int(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok, frame = cap.read()

        if not ok:
            print(f"Warning: could not read frame {frame_idx}")
            continue

        output_path = output_dir / f"frame_{i:02d}_{frame_idx}.jpg"
        cv2.imwrite(str(output_path), frame)

    cap.release()

    print(f"Extracted frames to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", default="outputs/inspection")
    parser.add_argument("--frames", type=int, default=12)

    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        args.frames,
    )


if __name__ == "__main__":
    main()