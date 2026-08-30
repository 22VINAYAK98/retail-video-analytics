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
OUTPUT_PATH = "outputs/ls_trajectory_analysis.mp4"

MODEL_PATH = "yolo26m.pt"

TRACKER_CONFIG = "botsort.yaml"
ROI_CONFIG = "configs/entrance.yaml"

CSV_OUTPUT = "outputs/ls_trajectory_metrics.csv"
PLOT_DIR = "outputs/ls_plots"

CONFIDENCE = 0.4

# Process at least the first 6000 frames
MAX_FRAMES = 6000

# ------------------------------------------------------------
# Sliding Least Squares window
# ------------------------------------------------------------
# This is specified in seconds rather than frames so that
# behavior remains consistent if FPS changes.
LS_WINDOW_SECONDS = 1.0

# Minimum samples required inside the LS window
MIN_LS_SAMPLES = 8

# Maximum displayed trajectory history
TRAIL_LENGTH = 300

# ------------------------------------------------------------
# Speed-trend window
# ------------------------------------------------------------
# We fit a local line to the estimated speed over this period.
SPEED_TREND_WINDOW_SECONDS = 1.0

# Minimum speed required before considering a slowdown.
# This prevents tiny/noisy movements from being classified
# as meaningful slowdown.
MIN_SPEED_FOR_SLOWDOWN = 8.0

# Negative speed slope threshold.
# Units: px/s^2
SPEED_SLOPE_THRESHOLD = -5.0

# Minimum duration for a sustained slowdown.
SLOWDOWN_MIN_DURATION_SECONDS = 0.5

# Minimum number of samples for plotting
MIN_TRACK_SAMPLES_FOR_PLOT = 20


# ============================================================
# ROI
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
    """Return ROI containing point."""

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
    Bottom-center of bounding box.

    Used as approximate ground/contact position
    of the person.
    """

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        y2
    )


# ============================================================
# Least Squares
# ============================================================

def linear_least_squares(times, values):
    """
    Fit:

        value = intercept + slope * time

    Returns:

        intercept
        slope

    The time values are centered before fitting to improve
    numerical stability.
    """

    times = np.asarray(
        times,
        dtype=np.float64
    )

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(times) < 2:
        return 0.0, 0.0

    t0 = times[-1]

    centered_time = times - t0

    A = np.column_stack(
        (
            np.ones(len(centered_time)),
            centered_time
        )
    )

    try:

        coefficients, _, _, _ = np.linalg.lstsq(
            A,
            values,
            rcond=None
        )

        intercept = coefficients[0]
        slope = coefficients[1]

        return (
            float(intercept),
            float(slope)
        )

    except np.linalg.LinAlgError:

        return 0.0, 0.0


# ============================================================
# Sliding trajectory fit
# ============================================================

def estimate_velocity_from_trajectory(
    position_history,
    current_time,
    window_seconds
):
    """
    Fit x(t) and y(t) independently using sliding-window
    Least Squares.

    Returns:

        smoothed_x
        smoothed_y
        vx
        vy
        speed
        direction
    """

    if len(position_history) < MIN_LS_SAMPLES:

        if len(position_history) == 0:
            return (
                None,
                None,
                0.0,
                0.0,
                0.0,
                None
            )

        # position_history stores:
        # (timestamp, x, y)

        _, x, y = position_history[-1]

        return (
            x,
            y,
            0.0,
            0.0,
            0.0,
            None
        )

    selected = []

    for timestamp, x, y in position_history:

        if (
            current_time - timestamp
            <= window_seconds
        ):

            selected.append(
                (
                    timestamp,
                    x,
                    y
                )
            )

    if len(selected) < MIN_LS_SAMPLES:

        selected = list(
            position_history
        )[-MIN_LS_SAMPLES:]

    times = np.array(
        [s[0] for s in selected],
        dtype=np.float64
    )

    xs = np.array(
        [s[1] for s in selected],
        dtype=np.float64
    )

    ys = np.array(
        [s[2] for s in selected],
        dtype=np.float64
    )

    _, vx = linear_least_squares(
        times,
        xs
    )

    _, vy = linear_least_squares(
        times,
        ys
    )

    # Evaluate fitted position at current time
    x_intercept, _ = linear_least_squares(
        times,
        xs
    )

    y_intercept, _ = linear_least_squares(
        times,
        ys
    )

    smoothed_x = x_intercept
    smoothed_y = y_intercept

    speed = np.sqrt(
        vx * vx +
        vy * vy
    )

    if speed > 1e-6:

        direction = np.degrees(
            np.arctan2(
                vy,
                vx
            )
        )

    else:

        direction = None

    return (
        smoothed_x,
        smoothed_y,
        float(vx),
        float(vy),
        float(speed),
        direction
    )


# ============================================================
# Speed trend
# ============================================================

def estimate_speed_trend(
    speed_history,
    current_time,
    window_seconds
):
    """
    Fit a local Least Squares line to speed.

        speed(t) = b0 + b1*t

    b1 is the speed trend:

        b1 > 0  -> accelerating
        b1 < 0  -> slowing down
        b1 ~ 0  -> approximately constant speed
    """

    selected = []

    for timestamp, speed in speed_history:

        if (
            current_time - timestamp
            <= window_seconds
        ):

            selected.append(
                (
                    timestamp,
                    speed
                )
            )

    if len(selected) < MIN_LS_SAMPLES:

        return 0.0

    times = np.array(
        [s[0] for s in selected],
        dtype=np.float64
    )

    speeds = np.array(
        [s[1] for s in selected],
        dtype=np.float64
    )

    _, slope = linear_least_squares(
        times,
        speeds
    )

    return float(slope)


# ============================================================
# Draw ROI
# ============================================================

def draw_rois(
    frame,
    rois
):

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
            (
                int(x),
                int(y) - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


# ============================================================
# Draw trajectory
# ============================================================

def draw_trajectory(
    frame,
    trajectory
):

    points = list(trajectory)

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
# Direction arrow
# ============================================================

def draw_direction_arrow(
    frame,
    position,
    direction,
    length=40
):

    if direction is None:
        return

    angle = np.radians(
        direction
    )

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


# ============================================================
# Plot
# ============================================================

def generate_track_plot(
    track_id,
    metrics,
    output_dir
):

    if len(metrics) < MIN_TRACK_SAMPLES_FOR_PLOT:
        return

    time = np.array(
        [m["time"] for m in metrics]
    )

    raw_x = np.array(
        [m["raw_x"] for m in metrics]
    )

    raw_y = np.array(
        [m["raw_y"] for m in metrics]
    )

    smooth_x = np.array(
        [m["smooth_x"] for m in metrics]
    )

    smooth_y = np.array(
        [m["smooth_y"] for m in metrics]
    )

    raw_speed = np.array(
        [m["raw_speed"] for m in metrics]
    )

    ls_speed = np.array(
        [m["speed"] for m in metrics]
    )

    speed_trend = np.array(
        [m["speed_trend"] for m in metrics]
    )

    slowdown = np.array(
        [m["slowdown"] for m in metrics]
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 15)
    )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    axes[0].plot(
        raw_x,
        raw_y,
        label="Raw trajectory",
        alpha=0.5
    )

    axes[0].plot(
        smooth_x,
        smooth_y,
        label="LS-smoothed trajectory",
        linewidth=2
    )

    axes[0].scatter(
        smooth_x[0],
        smooth_y[0],
        s=50,
        label="Start"
    )

    axes[0].scatter(
        smooth_x[-1],
        smooth_y[-1],
        s=50,
        label="End"
    )

    axes[0].invert_yaxis()

    axes[0].set_title(
        f"Track {track_id} - Trajectory"
    )

    axes[0].set_xlabel(
        "X (pixels)"
    )

    axes[0].set_ylabel(
        "Y (pixels)"
    )

    axes[0].legend()

    axes[0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    axes[1].plot(
        time,
        raw_speed,
        label="Raw speed",
        alpha=0.4
    )

    axes[1].plot(
        time,
        ls_speed,
        label="LS speed",
        linewidth=2
    )

    axes[1].set_title(
        f"Track {track_id} - Speed"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].set_ylabel(
        "Speed (px/s)"
    )

    axes[1].legend()

    axes[1].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Speed trend
    # --------------------------------------------------------

    axes[2].plot(
        time,
        speed_trend,
        label="LS speed trend"
    )

    axes[2].axhline(
        SPEED_SLOPE_THRESHOLD,
        linestyle="--",
        label="Slowdown threshold"
    )

    axes[2].axhline(
        0,
        linewidth=1
    )

    axes[2].set_title(
        f"Track {track_id} - Speed Trend"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].set_ylabel(
        "dv/dt (px/s²)"
    )

    axes[2].legend()

    axes[2].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Slowdown
    # --------------------------------------------------------

    axes[3].plot(
        time,
        ls_speed,
        label="LS speed"
    )

    slowdown_mask = slowdown.astype(bool)

    if np.any(slowdown_mask):

        axes[3].fill_between(
            time,
            0,
            ls_speed,
            where=slowdown_mask,
            alpha=0.3,
            label="Sustained slowdown"
        )

    axes[3].set_title(
        f"Track {track_id} - Slowdown Candidates"
    )

    axes[3].set_xlabel(
        "Time (s)"
    )

    axes[3].set_ylabel(
        "Speed (px/s)"
    )

    axes[3].legend()

    axes[3].grid(
        True,
        alpha=0.3
    )

    fig.suptitle(
        f"Least Squares Motion Analysis - Track {track_id}",
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

    fieldnames = [
        "track_id",
        "frame",
        "time_sec",
        "raw_x",
        "raw_y",
        "smooth_x",
        "smooth_y",
        "vx_px_s",
        "vy_px_s",
        "raw_speed_px_s",
        "speed_px_s",
        "direction_deg",
        "speed_trend_px_s2",
        "slowdown",
        "roi"
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

                    "track_id":
                        track_id,

                    "frame":
                        m["frame"],

                    "time_sec":
                        m["time"],

                    "raw_x":
                        m["raw_x"],

                    "raw_y":
                        m["raw_y"],

                    "smooth_x":
                        m["smooth_x"],

                    "smooth_y":
                        m["smooth_y"],

                    "vx_px_s":
                        m["vx"],

                    "vy_px_s":
                        m["vy"],

                    "raw_speed_px_s":
                        m["raw_speed"],

                    "speed_px_s":
                        m["speed"],

                    "direction_deg":
                        ""
                        if m["direction"] is None
                        else m["direction"],

                    "speed_trend_px_s2":
                        m["speed_trend"],

                    "slowdown":
                        int(m["slowdown"]),

                    "roi":
                        ""
                        if m["roi"] is None
                        else m["roi"]
                })


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Sliding Least Squares Trajectory Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load ROI
    # --------------------------------------------------------

    print("\nLoading ROI configuration...")

    rois = load_rois(
        ROI_CONFIG
    )

    for name in rois:

        print(
            f"  ROI: {name}"
        )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

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
        f"\nVideo: {VIDEO_PATH}"
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
        f"Frames to process: {frames_to_process}"
    )

    print(
        f"LS window: {LS_WINDOW_SECONDS:.2f} sec"
    )

    print(
        f"Speed trend window: "
        f"{SPEED_TREND_WINDOW_SECONDS:.2f} sec"
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
            "Could not create output video."
        )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    position_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    trajectory_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    speed_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Complete track metrics
    all_metrics = defaultdict(list)

    # --------------------------------------------------------
    # Slowdown state
    # --------------------------------------------------------

    slowdown_start_time = {}

    slowdown_duration = defaultdict(
        float
    )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number > frames_to_process:
            break

        current_time = (
            frame_number / fps
        )

        output_frame = frame.copy()

        draw_rois(
            output_frame,
            rois
        )

        # ----------------------------------------------------
        # YOLO + BoT-SORT
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

        result = results[0]

        active_tracks = 0

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

                # --------------------------------------------
                # Raw position
                # --------------------------------------------

                raw_x, raw_y = bottom_center(
                    box
                )

                # --------------------------------------------
                # Save position
                # --------------------------------------------

                position_history[
                    track_id
                ].append(
                    (
                        current_time,
                        raw_x,
                        raw_y
                    )
                )

                # --------------------------------------------
                # Raw instantaneous speed
                # --------------------------------------------

                raw_speed = 0.0

                history = position_history[
                    track_id
                ]

                if len(history) >= 2:

                    t1, x1, y1 = history[-2]
                    t2, x2, y2 = history[-1]

                    dt = t2 - t1

                    if dt > 0:

                        distance = np.sqrt(
                            (x2 - x1) ** 2 +
                            (y2 - y1) ** 2
                        )

                        raw_speed = (
                            distance / dt
                        )

                # --------------------------------------------
                # Least Squares velocity
                # --------------------------------------------

                (
                    smooth_x,
                    smooth_y,
                    vx,
                    vy,
                    speed,
                    direction
                ) = estimate_velocity_from_trajectory(
                    position_history[
                        track_id
                    ],
                    current_time,
                    LS_WINDOW_SECONDS
                )

                # --------------------------------------------
                # Store trajectory
                # --------------------------------------------

                trajectory_history[
                    track_id
                ].append(
                    (
                        smooth_x,
                        smooth_y
                    )
                )

                # --------------------------------------------
                # Speed history
                # --------------------------------------------

                speed_history[
                    track_id
                ].append(
                    (
                        current_time,
                        speed
                    )
                )

                # --------------------------------------------
                # Speed trend
                # --------------------------------------------

                speed_trend = (
                    estimate_speed_trend(
                        speed_history[
                            track_id
                        ],
                        current_time,
                        SPEED_TREND_WINDOW_SECONDS
                    )
                )

                # --------------------------------------------
                # ROI
                # --------------------------------------------

                roi = get_roi(
                    (
                        raw_x,
                        raw_y
                    ),
                    rois
                )

                # --------------------------------------------
                # Slowdown candidate
                # --------------------------------------------

                slowdown_candidate = (
                    speed >= MIN_SPEED_FOR_SLOWDOWN
                    and
                    speed_trend
                    <= SPEED_SLOPE_THRESHOLD
                )

                slowdown = False

                if slowdown_candidate:

                    if track_id not in slowdown_start_time:

                        slowdown_start_time[
                            track_id
                        ] = current_time

                    duration = (
                        current_time
                        -
                        slowdown_start_time[
                            track_id
                        ]
                    )

                    slowdown_duration[
                        track_id
                    ] = duration

                    if (
                        duration
                        >=
                        SLOWDOWN_MIN_DURATION_SECONDS
                    ):

                        slowdown = True

                else:

                    slowdown_start_time.pop(
                        track_id,
                        None
                    )

                    slowdown_duration[
                        track_id
                    ] = 0.0

                # --------------------------------------------
                # Save metrics
                # --------------------------------------------

                all_metrics[
                    track_id
                ].append({

                    "frame":
                        frame_number,

                    "time":
                        current_time,

                    "raw_x":
                        raw_x,

                    "raw_y":
                        raw_y,

                    "smooth_x":
                        smooth_x,

                    "smooth_y":
                        smooth_y,

                    "vx":
                        vx,

                    "vy":
                        vy,

                    "raw_speed":
                        raw_speed,

                    "speed":
                        speed,

                    "direction":
                        direction,

                    "speed_trend":
                        speed_trend,

                    "slowdown":
                        slowdown,

                    "roi":
                        roi
                })

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

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

                # --------------------------------------------
                # Trajectory
                # --------------------------------------------

                draw_trajectory(
                    output_frame,
                    trajectory_history[
                        track_id
                    ]
                )

                # --------------------------------------------
                # Direction
                # --------------------------------------------

                draw_direction_arrow(
                    output_frame,
                    (
                        smooth_x,
                        smooth_y
                    ),
                    direction
                )

                # --------------------------------------------
                # ID
                # --------------------------------------------

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

                # --------------------------------------------
                # LS speed
                # --------------------------------------------

                cv2.putText(
                    output_frame,
                    f"LS Speed: {speed:.1f} px/s",
                    (
                        x1,
                        y2 + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                # --------------------------------------------
                # Speed trend
                # --------------------------------------------

                cv2.putText(
                    output_frame,
                    f"Speed Trend: {speed_trend:.1f}",
                    (
                        x1,
                        y2 + 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # Slowdown state
                # --------------------------------------------

                if slowdown:

                    label = (
                        f"SLOWDOWN "
                        f"{slowdown_duration[track_id]:.1f}s"
                    )

                    cv2.putText(
                        output_frame,
                        label,
                        (
                            x1,
                            y2 + 60
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2
                    )

                # --------------------------------------------
                # ROI
                # --------------------------------------------

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

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        cv2.putText(
            output_frame,
            (
                f"Frame: "
                f"{frame_number}/"
                f"{frames_to_process}"
            ),
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
            (
                f"LS Window: "
                f"{LS_WINDOW_SECONDS:.1f}s"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        writer.write(
            output_frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{frames_to_process}"
                f" | Tracks: "
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

    # ========================================================
    # Plots
    # ========================================================

    print("\nGenerating plots...")

    plot_count = 0

    for track_id, metrics in all_metrics.items():

        if len(metrics) < MIN_TRACK_SAMPLES_FOR_PLOT:
            continue

        generate_track_plot(
            track_id,
            metrics,
            PLOT_DIR
        )

        plot_count += 1

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("Least Squares trajectory analysis completed")
    print("=" * 70)

    print(
        f"Frames processed : {frame_number}"
    )

    print(
        f"Unique tracks    : {len(all_metrics)}"
    )

    print(
        f"Plots generated  : {plot_count}"
    )

    print(
        f"Output video     : {OUTPUT_PATH}"
    )

    print(
        f"CSV              : {CSV_OUTPUT}"
    )

    print(
        f"Plots            : {PLOT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()