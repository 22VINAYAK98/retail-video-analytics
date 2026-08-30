import os
import csv
import cv2
import yaml
import numpy as np

from collections import defaultdict, deque
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

VIDEO_PATH = "data/entrance.mp4"
OUTPUT_PATH = "outputs/trajectory_analysis.mp4"

MODEL_PATH = "yolo26m.pt"
TRACKER_CONFIG = "botsort.yaml"
ROI_CONFIG = "configs/entrance.yaml"

CSV_OUTPUT = "outputs/trajectory_metrics.csv"
PLOT_DIR = "outputs/plots"

CONFIDENCE = 0.4

# Process first 6000 frames
MAX_FRAMES = 6000

# Displayed trajectory length
TRAIL_LENGTH = 300

# Temporal window for speed / acceleration
MOTION_WINDOW_FRAMES = 5

# Position smoothing
POSITION_SMOOTHING = 0.2

# Speed smoothing
SPEED_SMOOTHING = 0.3

# Acceleration smoothing
ACCELERATION_SMOOTHING = 0.3

# Minimum number of samples required
# before generating an individual plot
MIN_TRACK_SAMPLES_FOR_PLOT = 20


# ============================================================
# ROI Functions
# ============================================================

def load_rois(path):
    """Load ROI polygons from YAML."""

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    rois = {}

    for name, roi in data["rois"].items():

        rois[name] = np.array(
            roi["polygon"],
            dtype=np.int32
        )

    return rois


def get_roi(point, rois):
    """Return the ROI containing the point."""

    x, y = point

    for name, polygon in rois.items():

        inside = cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        )

        if inside >= 0:
            return name

    return None


# ============================================================
# Position
# ============================================================

def bottom_center(box):
    """
    Return bottom-center of bounding box.

    This approximates the person's ground/contact point.
    """

    x1, y1, x2, y2 = box

    x = (x1 + x2) / 2.0
    y = y2

    return x, y


# ============================================================
# Temporal History Helpers
# ============================================================

def get_history_sample(
    history,
    target_frame
):
    """
    Return the latest history sample whose frame
    is <= target_frame.

    History format:

        (frame_number, value)
    """

    previous_sample = None

    for frame_idx, value in history:

        if frame_idx <= target_frame:

            previous_sample = (
                frame_idx,
                value
            )

        else:
            break

    return previous_sample


# ============================================================
# Speed
# ============================================================

def calculate_windowed_speed(
    position_history,
    current_frame,
    fps,
    window_frames
):
    """
    Calculate speed over a temporal window.

    speed = displacement / elapsed_time
    """

    if len(position_history) < 2:
        return 0.0

    current_frame_idx, current_position = (
        position_history[-1]
    )

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = get_history_sample(
        position_history,
        target_frame
    )

    if previous_sample is None:
        return 0.0

    previous_frame, previous_position = (
        previous_sample
    )

    p1 = np.array(
        previous_position,
        dtype=np.float32
    )

    p2 = np.array(
        current_position,
        dtype=np.float32
    )

    displacement = p2 - p1

    distance = np.linalg.norm(
        displacement
    )

    frame_delta = (
        current_frame_idx
        - previous_frame
    )

    if frame_delta <= 0:
        return 0.0

    dt = frame_delta / fps

    return float(
        distance / dt
    )


# ============================================================
# Direction
# ============================================================

def calculate_direction(
    position_history,
    current_frame,
    window_frames
):
    """
    Calculate movement direction over the temporal window.
    """

    if len(position_history) < 2:
        return None

    current_frame_idx, current_position = (
        position_history[-1]
    )

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = get_history_sample(
        position_history,
        target_frame
    )

    if previous_sample is None:
        return None

    _, previous_position = previous_sample

    p1 = np.array(
        previous_position,
        dtype=np.float32
    )

    p2 = np.array(
        current_position,
        dtype=np.float32
    )

    displacement = p2 - p1

    if np.linalg.norm(displacement) < 1e-6:
        return None

    dx = displacement[0]
    dy = displacement[1]

    return float(
        np.degrees(
            np.arctan2(dy, dx)
        )
    )


# ============================================================
# Acceleration
# ============================================================

def calculate_windowed_acceleration(
    speed_history,
    current_frame,
    fps,
    window_frames
):
    """
    Calculate acceleration over a temporal window.

    acceleration =
        (current_speed - previous_speed) / dt
    """

    if len(speed_history) < 2:
        return 0.0

    current_frame_idx, current_speed = (
        speed_history[-1]
    )

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = get_history_sample(
        speed_history,
        target_frame
    )

    if previous_sample is None:
        return 0.0

    previous_frame, previous_speed = (
        previous_sample
    )

    frame_delta = (
        current_frame_idx
        - previous_frame
    )

    if frame_delta <= 0:
        return 0.0

    dt = frame_delta / fps

    return float(
        (current_speed - previous_speed)
        / dt
    )


# ============================================================
# Drawing
# ============================================================

def draw_rois(frame, rois):

    for name, polygon in rois.items():

        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2
        )

        x, y = polygon[0]

        cv2.putText(
            frame,
            name,
            (int(x), int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


def draw_direction_arrow(
    frame,
    position,
    direction,
    length=40
):

    if direction is None:
        return

    angle = np.radians(direction)

    dx = int(
        np.cos(angle) * length
    )

    dy = int(
        np.sin(angle) * length
    )

    start = (
        int(position[0]),
        int(position[1])
    )

    end = (
        int(position[0] + dx),
        int(position[1] + dy)
    )

    cv2.arrowedLine(
        frame,
        start,
        end,
        (0, 255, 255),
        2,
        tipLength=0.25
    )


def draw_trajectory(
    frame,
    history
):

    points = list(history)

    if len(points) < 2:
        return

    for i in range(1, len(points)):

        p1 = (
            int(points[i - 1][0]),
            int(points[i - 1][1])
        )

        p2 = (
            int(points[i][0]),
            int(points[i][1])
        )

        cv2.line(
            frame,
            p1,
            p2,
            (0, 0, 255),
            2
        )


# ============================================================
# Plot Generation
# ============================================================

def generate_track_plot(
    track_id,
    metrics,
    fps,
    output_dir
):
    """
    Generate one diagnostic plot per person.

    Contains:

        1. Speed
        2. Acceleration
        3. Deceleration
        4. X/Y trajectory
    """

    if len(metrics) < MIN_TRACK_SAMPLES_FOR_PLOT:
        return

    frames = np.array(
        [m["frame"] for m in metrics]
    )

    time = frames / fps

    speed = np.array(
        [m["speed"] for m in metrics]
    )

    acceleration = np.array(
        [m["acceleration"] for m in metrics]
    )

    deceleration = np.array(
        [m["deceleration"] for m in metrics]
    )

    x = np.array(
        [m["x"] for m in metrics]
    )

    y = np.array(
        [m["y"] for m in metrics]
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 14)
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    axes[0].plot(
        time,
        speed
    )

    axes[0].set_title(
        f"Track ID {track_id} - Speed"
    )

    axes[0].set_ylabel(
        "Speed (px/s)"
    )

    axes[0].set_xlabel(
        "Time (s)"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    axes[1].plot(
        time,
        acceleration
    )

    axes[1].axhline(
        0,
        linewidth=1
    )

    axes[1].set_title(
        f"Track ID {track_id} - Acceleration"
    )

    axes[1].set_ylabel(
        "Acceleration (px/s²)"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Deceleration
    # --------------------------------------------------------

    axes[2].plot(
        time,
        deceleration
    )

    axes[2].set_title(
        f"Track ID {track_id} - Deceleration"
    )

    axes[2].set_ylabel(
        "Deceleration (px/s²)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    axes[3].plot(
        x,
        y
    )

    axes[3].scatter(
        x[0],
        y[0],
        s=50,
        label="Start"
    )

    axes[3].scatter(
        x[-1],
        y[-1],
        s=50,
        label="End"
    )

    axes[3].invert_yaxis()

    axes[3].set_title(
        f"Track ID {track_id} - Trajectory"
    )

    axes[3].set_xlabel(
        "X (pixels)"
    )

    axes[3].set_ylabel(
        "Y (pixels)"
    )

    axes[3].legend()

    axes[3].grid(
        True,
        alpha=0.3
    )

    fig.suptitle(
        f"Trajectory Analysis - Track {track_id}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"ID_{track_id}.png"
    )

    fig.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# CSV
# ============================================================

def write_csv(
    all_metrics,
    output_path
):
    """Write all trajectory metrics to CSV."""

    fieldnames = [
        "track_id",
        "frame",
        "time_sec",
        "x",
        "y",
        "roi",
        "speed_px_s",
        "direction_deg",
        "acceleration_px_s2",
        "deceleration_px_s2"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for track_id, metrics in all_metrics.items():

            for m in metrics:

                writer.writerow({
                    "track_id": track_id,
                    "frame": m["frame"],
                    "time_sec": m["time"],
                    "x": m["x"],
                    "y": m["y"],
                    "roi": m["roi"] or "",
                    "speed_px_s": m["speed"],
                    "direction_deg": (
                        ""
                        if m["direction"] is None
                        else m["direction"]
                    ),
                    "acceleration_px_s2":
                        m["acceleration"],
                    "deceleration_px_s2":
                        m["deceleration"]
                })


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Trajectory Analysis + Motion Visualization")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load ROIs
    # --------------------------------------------------------

    print("\nLoading ROIs...")

    rois = load_rois(
        ROI_CONFIG
    )

    for name in rois:

        print(
            f"  - {name}"
        )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    print("\nOpening video...")

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frames_to_process = min(
        MAX_FRAMES,
        total_frames
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Processing: {frames_to_process}"
    )

    print(
        f"Motion window: "
        f"{MOTION_WINDOW_FRAMES} frames"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Could not create output: "
            f"{OUTPUT_PATH}"
        )

    # ========================================================
    # Per-person state
    # ========================================================

    # Smoothed current position
    smoothed_positions = {}

    # Display trajectory
    trajectories = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Frame-aware position history
    position_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Frame-aware speed history
    speed_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Smoothed speed
    smoothed_speed = {}

    # Smoothed acceleration
    smoothed_acceleration = {}

    # Complete metrics for CSV/plots
    all_metrics = defaultdict(list)

    # Current ROI
    current_rois = {}

    # ========================================================
    # Process video
    # ========================================================

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number > MAX_FRAMES:
            break

        # ----------------------------------------------------
        # Detection + Tracking
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            tracker=TRACKER_CONFIG,
            verbose=False,
            device=0
        )

        output_frame = frame.copy()

        # ----------------------------------------------------
        # Draw ROIs
        # ----------------------------------------------------

        draw_rois(
            output_frame,
            rois
        )

        result = results[0]

        active_tracks = 0

        # ----------------------------------------------------
        # Tracked persons
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            active_tracks = len(
                track_ids
            )

            for box, track_id in zip(
                boxes,
                track_ids
            ):

                # =================================================
                # Raw position
                # =================================================

                raw_position = bottom_center(
                    box
                )

                raw_position_array = np.array(
                    raw_position,
                    dtype=np.float32
                )

                # =================================================
                # Position smoothing
                # =================================================

                if track_id not in smoothed_positions:

                    smoothed_positions[
                        track_id
                    ] = raw_position_array

                else:

                    previous_position = (
                        smoothed_positions[
                            track_id
                        ]
                    )

                    smoothed_positions[
                        track_id
                    ] = (
                        POSITION_SMOOTHING
                        * raw_position_array
                        +
                        (
                            1.0
                            - POSITION_SMOOTHING
                        )
                        * previous_position
                    )

                smoothed_position = (
                    smoothed_positions[
                        track_id
                    ]
                )

                smoothed_point = (
                    float(smoothed_position[0]),
                    float(smoothed_position[1])
                )

                # =================================================
                # Position history
                # =================================================

                position_history[
                    track_id
                ].append(
                    (
                        frame_number,
                        smoothed_point
                    )
                )

                trajectories[
                    track_id
                ].append(
                    smoothed_point
                )

                # =================================================
                # Speed
                # =================================================

                raw_speed = (
                    calculate_windowed_speed(
                        position_history[
                            track_id
                        ],
                        frame_number,
                        fps,
                        MOTION_WINDOW_FRAMES
                    )
                )

                if track_id not in smoothed_speed:

                    smoothed_speed[
                        track_id
                    ] = raw_speed

                else:

                    smoothed_speed[
                        track_id
                    ] = (
                        SPEED_SMOOTHING
                        * raw_speed
                        +
                        (
                            1.0
                            - SPEED_SMOOTHING
                        )
                        * smoothed_speed[
                            track_id
                        ]
                    )

                speed = smoothed_speed[
                    track_id
                ]

                # =================================================
                # Speed history
                # =================================================

                speed_history[
                    track_id
                ].append(
                    (
                        frame_number,
                        speed
                    )
                )

                # =================================================
                # Acceleration
                # =================================================

                raw_acceleration = (
                    calculate_windowed_acceleration(
                        speed_history[
                            track_id
                        ],
                        frame_number,
                        fps,
                        MOTION_WINDOW_FRAMES
                    )
                )

                if (
                    track_id
                    not in smoothed_acceleration
                ):

                    smoothed_acceleration[
                        track_id
                    ] = raw_acceleration

                else:

                    smoothed_acceleration[
                        track_id
                    ] = (
                        ACCELERATION_SMOOTHING
                        * raw_acceleration
                        +
                        (
                            1.0
                            - ACCELERATION_SMOOTHING
                        )
                        * smoothed_acceleration[
                            track_id
                        ]
                    )

                acceleration = (
                    smoothed_acceleration[
                        track_id
                    ]
                )

                # =================================================
                # Deceleration
                # =================================================

                deceleration = max(
                    0.0,
                    -acceleration
                )

                # =================================================
                # Direction
                # =================================================

                direction = calculate_direction(
                    position_history[
                        track_id
                    ],
                    frame_number,
                    MOTION_WINDOW_FRAMES
                )

                # =================================================
                # ROI
                # =================================================

                roi = get_roi(
                    raw_position,
                    rois
                )

                current_rois[
                    track_id
                ] = roi

                # =================================================
                # Store metrics
                # =================================================

                all_metrics[
                    track_id
                ].append({
                    "frame": frame_number,
                    "time": frame_number / fps,
                    "x": smoothed_point[0],
                    "y": smoothed_point[1],
                    "roi": roi,
                    "speed": speed,
                    "direction": direction,
                    "acceleration": acceleration,
                    "deceleration": deceleration
                })

                # =================================================
                # Bounding box
                # =================================================

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # =================================================
                # Trajectory
                # =================================================

                draw_trajectory(
                    output_frame,
                    trajectories[
                        track_id
                    ]
                )

                # =================================================
                # Direction
                # =================================================

                draw_direction_arrow(
                    output_frame,
                    smoothed_point,
                    direction
                )

                # =================================================
                # ID
                # =================================================

                cv2.putText(
                    output_frame,
                    f"ID {track_id}",
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # =================================================
                # Speed
                # =================================================

                cv2.putText(
                    output_frame,
                    f"Speed: {speed:.1f} px/s",
                    (
                        x1,
                        y2 + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                # =================================================
                # Acceleration
                # =================================================

                cv2.putText(
                    output_frame,
                    f"Accel: {acceleration:.1f} px/s2",
                    (
                        x1,
                        y2 + 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                # =================================================
                # Deceleration
                # =================================================

                cv2.putText(
                    output_frame,
                    f"Decel: {deceleration:.1f} px/s2",
                    (
                        x1,
                        y2 + 60
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

                # =================================================
                # ROI
                # =================================================

                if roi is not None:

                    cv2.putText(
                        output_frame,
                        roi,
                        (
                            x1,
                            y2 + 80
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

        # --------------------------------------------------------
        # Frame info
        # --------------------------------------------------------

        cv2.putText(
            output_frame,
            f"Frame: {frame_number}/{frames_to_process}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Active tracks: {active_tracks}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Motion window: {MOTION_WINDOW_FRAMES} frames",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # --------------------------------------------------------
        # Write video
        # --------------------------------------------------------

        writer.write(
            output_frame
        )

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{frames_to_process} "
                f"| Tracks: "
                f"{len(all_metrics)}"
            )

    # ========================================================
    # Cleanup
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # CSV
    # ========================================================

    print("\nWriting CSV...")

    write_csv(
        all_metrics,
        CSV_OUTPUT
    )

    print(
        f"CSV written: {CSV_OUTPUT}"
    )

    # ========================================================
    # Plots
    # ========================================================

    print("\nGenerating plots...")

    generated_plots = 0

    for track_id, metrics in all_metrics.items():

        if len(metrics) < MIN_TRACK_SAMPLES_FOR_PLOT:
            continue

        generate_track_plot(
            track_id,
            metrics,
            fps,
            PLOT_DIR
        )

        generated_plots += 1

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("Trajectory analysis completed.")
    print("=" * 70)

    print(
        f"Frames processed: {frame_number}"
    )

    print(
        f"Unique tracks: {len(all_metrics)}"
    )

    print(
        f"Plots generated: {generated_plots}"
    )

    print(
        f"Video: {OUTPUT_PATH}"
    )

    print(
        f"CSV: {CSV_OUTPUT}"
    )

    print(
        f"Plots: {PLOT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()