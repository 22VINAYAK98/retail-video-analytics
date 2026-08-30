import cv2
import numpy as np
import yaml

from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

VIDEO_PATH = "data/entrance.mp4"
OUTPUT_PATH = "outputs/trajectory_analysis.mp4"

MODEL_PATH = "yolo26m.pt"
TRACKER_CONFIG = "botsort.yaml"
ROI_CONFIG = "configs/entrance.yaml"

CONFIDENCE = 0.4

# Process first 6000 frames
MAX_FRAMES = 6000

# Number of trajectory points displayed
TRAIL_LENGTH = 300

# Temporal window used for motion estimation
MOTION_WINDOW_FRAMES = 5

# Position smoothing
POSITION_SMOOTHING = 0.2

# Speed smoothing
SPEED_SMOOTHING = 0.3

# Acceleration smoothing
ACCELERATION_SMOOTHING = 0.3


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
# Position Functions
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
# Motion Functions
# ============================================================

def calculate_windowed_speed(
    position_history,
    current_frame,
    fps,
    window_frames
):
    """
    Calculate speed using displacement over a temporal window.

    Instead of comparing consecutive frames:

        P(t) - P(t-1)

    we compare:

        P(t) - P(t-window)

    This greatly reduces frame-to-frame tracking noise.
    """

    if len(position_history) < 2:
        return 0.0

    current_position = np.array(
        position_history[-1][1],
        dtype=np.float32
    )

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = None

    # Find the oldest point that is close to
    # the requested temporal window.
    for frame_idx, position in position_history:

        if frame_idx <= target_frame:
            previous_sample = (
                frame_idx,
                position
            )
        else:
            break

    if previous_sample is None:

        # Not enough temporal history yet.
        return 0.0

    previous_frame, previous_position = (
        previous_sample
    )

    previous_position = np.array(
        previous_position,
        dtype=np.float32
    )

    displacement = (
        current_position
        - previous_position
    )

    distance = np.linalg.norm(
        displacement
    )

    frame_delta = (
        current_frame
        - previous_frame
    )

    if frame_delta <= 0:
        return 0.0

    dt = frame_delta / fps

    speed = distance / dt

    return float(speed)


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

    current_position = np.array(
        position_history[-1][1],
        dtype=np.float32
    )

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = None

    for frame_idx, position in position_history:

        if frame_idx <= target_frame:
            previous_sample = (
                frame_idx,
                position
            )
        else:
            break

    if previous_sample is None:
        return None

    previous_position = np.array(
        previous_sample[1],
        dtype=np.float32
    )

    displacement = (
        current_position
        - previous_position
    )

    if np.linalg.norm(displacement) < 1e-6:
        return None

    dx = displacement[0]
    dy = displacement[1]

    direction = np.degrees(
        np.arctan2(dy, dx)
    )

    return float(direction)


def calculate_windowed_acceleration(
    speed_history,
    current_frame,
    fps,
    window_frames
):
    """
    Calculate acceleration using speed values separated
    by a temporal window.

        acceleration =
            (current_speed - previous_speed) / dt

    The speed itself is already temporally averaged,
    making this much less sensitive to frame-level jitter.
    """

    if len(speed_history) < 2:
        return 0.0

    current_speed = speed_history[-1][1]

    target_frame = (
        current_frame - window_frames
    )

    previous_sample = None

    for frame_idx, speed in speed_history:

        if frame_idx <= target_frame:
            previous_sample = (
                frame_idx,
                speed
            )
        else:
            break

    if previous_sample is None:
        return 0.0

    previous_frame, previous_speed = (
        previous_sample
    )

    frame_delta = (
        current_frame
        - previous_frame
    )

    if frame_delta <= 0:
        return 0.0

    dt = frame_delta / fps

    acceleration = (
        current_speed
        - previous_speed
    ) / dt

    return float(acceleration)


# ============================================================
# Drawing Functions
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
    """Draw smoothed trajectory."""

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
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Trajectory Analyzer")
    print("=" * 60)

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

    print(
        f"Loaded {len(rois)} ROIs:"
    )

    for name in rois:
        print(f"  - {name}")

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

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    frames_to_process = min(
        MAX_FRAMES,
        total_frames
    )

    print(
        f"Processing frames: "
        f"{frames_to_process}"
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

    # Smoothed trajectory displayed on video
    trajectories = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Position history:
    #
    # (frame_number, smoothed_position)
    #
    position_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Current smoothed position
    smoothed_positions = {}

    # Speed history:
    #
    # (frame_number, smoothed_speed)
    #
    speed_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Smoothed speed
    smoothed_speed = {}

    # Smoothed acceleration
    smoothed_acceleration = {}

    # Current ROI
    current_rois = {}

    # ========================================================
    # Frame processing
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

        active_track_count = 0

        # ----------------------------------------------------
        # Process tracked persons
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

            active_track_count = len(
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
                    int(smoothed_position[0]),
                    int(smoothed_position[1])
                )

                # =================================================
                # Store trajectory
                # =================================================

                trajectories[
                    track_id
                ].append(
                    smoothed_point
                )

                # Store frame-aware position history
                position_history[
                    track_id
                ].append(
                    (
                        frame_number,
                        smoothed_point
                    )
                )

                # =================================================
                # Calculate windowed speed
                # =================================================

                raw_speed = calculate_windowed_speed(
                    position_history[
                        track_id
                    ],
                    frame_number,
                    fps,
                    MOTION_WINDOW_FRAMES
                )

                # =================================================
                # Smooth speed
                # =================================================

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

                # Store speed with frame number
                speed_history[
                    track_id
                ].append(
                    (
                        frame_number,
                        speed
                    )
                )

                # =================================================
                # Calculate windowed acceleration
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

                # =================================================
                # Smooth acceleration
                # =================================================

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
                #
                # Raw position is deliberately used for ROI
                # membership so smoothing does not move a person
                # across a boundary artificially.
                # =================================================

                roi = get_roi(
                    raw_position,
                    rois
                )

                current_rois[
                    track_id
                ] = roi

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
                # Draw trajectory
                # =================================================

                draw_trajectory(
                    output_frame,
                    trajectories[
                        track_id
                    ]
                )

                # =================================================
                # Draw direction
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
        # Frame information
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
            f"Active tracks: {active_track_count}",
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
        # Write output
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
                f"{frames_to_process}"
            )

    # ========================================================
    # Cleanup
    # ========================================================

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("Trajectory analysis completed.")
    print("=" * 60)

    print(
        f"Output: {OUTPUT_PATH}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()