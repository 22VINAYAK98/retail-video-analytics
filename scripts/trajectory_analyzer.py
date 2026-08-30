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
MAX_FRAMES = 1000

# Number of trajectory points displayed per person
TRAIL_LENGTH = 300

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
    """Return the ROI containing the point, or None."""

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
# Position / Motion Functions
# ============================================================

def bottom_center(box):
    """
    Calculate the bottom-center point of a bounding box.

    This approximates the person's ground/contact position.
    """

    x1, y1, x2, y2 = box

    x = (x1 + x2) / 2.0
    y = y2

    return x, y


def calculate_motion(history, fps):
    """
    Calculate instantaneous speed and direction from
    the last two smoothed trajectory points.
    """

    if len(history) < 2:
        return 0.0, None

    p1 = np.array(
        history[-2],
        dtype=np.float32
    )

    p2 = np.array(
        history[-1],
        dtype=np.float32
    )

    displacement = p2 - p1

    distance = np.linalg.norm(
        displacement
    )

    speed = distance * fps

    dx = displacement[0]
    dy = displacement[1]

    direction = np.degrees(
        np.arctan2(dy, dx)
    )

    return speed, direction


# ============================================================
# Drawing Functions
# ============================================================

def draw_rois(frame, rois):
    """Draw configured ROIs on the frame."""

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
    """Draw the current movement direction."""

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


def draw_trajectory(
    frame,
    history
):
    """Draw the smoothed trajectory."""

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

    # Smoothed trajectory history
    trajectories = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Current smoothed position
    smoothed_positions = {}

    # Smoothed speed
    smoothed_speed = {}

    # Previous speed
    previous_speed = {}

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
                # Store smoothed trajectory
                # =================================================

                trajectories[
                    track_id
                ].append(
                    smoothed_point
                )

                history = trajectories[
                    track_id
                ]

                # =================================================
                # Calculate speed and direction
                # =================================================

                raw_speed, direction = (
                    calculate_motion(
                        history,
                        fps
                    )
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

                # =================================================
                # Calculate acceleration
                # =================================================

                dt = 1.0 / fps

                if track_id not in previous_speed:

                    acceleration = 0.0

                else:

                    acceleration = (
                        speed
                        - previous_speed[
                            track_id
                        ]
                    ) / dt

                previous_speed[
                    track_id
                ] = speed

                # =================================================
                # Smooth acceleration
                # =================================================

                if (
                    track_id
                    not in smoothed_acceleration
                ):

                    smoothed_acceleration[
                        track_id
                    ] = acceleration

                else:

                    smoothed_acceleration[
                        track_id
                    ] = (
                        ACCELERATION_SMOOTHING
                        * acceleration
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
                # Calculate deceleration
                # =================================================

                deceleration = max(
                    0.0,
                    -acceleration
                )

                # =================================================
                # ROI
                #
                # Raw position is used deliberately.
                # Smoothing should not alter ROI membership.
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
                    history
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
                # Draw ID
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
                # Draw speed
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
                # Draw acceleration
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
                # Draw deceleration
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
                # Draw ROI
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

# import cv2
# import numpy as np
# import yaml

# from collections import defaultdict, deque
# from ultralytics import YOLO


# # ============================================================
# # Configuration
# # ============================================================

# VIDEO_PATH = "data/entrance.mp4"
# OUTPUT_PATH = "outputs/trajectory_analysis.mp4"

# MODEL_PATH = "yolo26m.pt"
# TRACKER_CONFIG = "botsort.yaml"
# ROI_CONFIG = "configs/entrance.yaml"

# CONFIDENCE = 0.4

# # Process first 6000 frames
# MAX_FRAMES = 2000

# # Number of trajectory points displayed per person
# TRAIL_LENGTH = 300

# # Position smoothing
# #
# # Smaller value = smoother but slower response
# # Larger value = more responsive but noisier
# POSITION_SMOOTHING = 0.2

# # Speed smoothing
# SPEED_SMOOTHING = 0.3


# # ============================================================
# # ROI Functions
# # ============================================================

# def load_rois(path):
#     """Load ROI polygons from YAML."""

#     with open(path, "r") as f:
#         data = yaml.safe_load(f)

#     rois = {}

#     for name, roi in data["rois"].items():

#         rois[name] = np.array(
#             roi["polygon"],
#             dtype=np.int32
#         )

#     return rois


# def get_roi(point, rois):
#     """
#     Determine which ROI contains a point.

#     Returns:
#         ROI name or None
#     """

#     x, y = point

#     for name, polygon in rois.items():

#         inside = cv2.pointPolygonTest(
#             polygon,
#             (float(x), float(y)),
#             False
#         )

#         if inside >= 0:
#             return name

#     return None


# # ============================================================
# # Position / Motion Functions
# # ============================================================

# def bottom_center(box):
#     """
#     Return bottom-center of bounding box.

#     This is used as an approximation of the
#     person's ground position.
#     """

#     x1, y1, x2, y2 = box

#     x = (x1 + x2) / 2
#     y = y2

#     return x, y


# def calculate_motion(history, fps):
#     """
#     Calculate speed and direction from the
#     last two smoothed trajectory positions.
#     """

#     if len(history) < 2:
#         return 0.0, None

#     p1 = np.array(
#         history[-2],
#         dtype=np.float32
#     )

#     p2 = np.array(
#         history[-1],
#         dtype=np.float32
#     )

#     displacement = p2 - p1

#     distance = np.linalg.norm(
#         displacement
#     )

#     speed = distance * fps

#     dx = displacement[0]
#     dy = displacement[1]

#     direction = np.degrees(
#         np.arctan2(dy, dx)
#     )

#     return speed, direction


# # ============================================================
# # Drawing Functions
# # ============================================================

# def draw_rois(frame, rois):
#     """Draw all configured ROIs."""

#     for name, polygon in rois.items():

#         cv2.polylines(
#             frame,
#             [polygon],
#             isClosed=True,
#             color=(255, 255, 0),
#             thickness=2TRACKER_CONFIG = "botsort.yaml"
# ROI_CONFIG = "configs/entrance.yaml"
#         )

#         x, y = polygon[0]

#         cv2.putText(
#             frame,
#             name,
#             (int(x), int(y) - 8),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (255, 255, 0),
#             2
#         )


# def draw_direction_arrow(
#     frame,
#     position,
#     direction,
#     length=40
# ):
#     """Draw movement direction arrow."""

#     if direction is None:
#         return

#     angle = np.radians(direction)

#     dx = int(
#         np.cos(angle) * length
#     )

#     dy = int(
#         np.sin(angle) * length
#     )

#     start = (
#         int(position[0]),
#         int(position[1])
#     )

#     end = (
#         int(position[0] + dx),
#         int(position[1] + dy)
#     )

#     cv2.arrowedLine(
#         frame,
#         start,
#         end,
#         (0, 255, 255),
#         2,
#         tipLength=0.25
#     )


# def draw_trajectory(
#     frame,
#     history
# ):
#     """Draw smoothed trajectory."""

#     points = list(history)

#     if len(points) < 2:
#         return

#     for i in range(1, len(points)):

#         p1 = (
#             int(points[i - 1][0]),
#             int(points[i - 1][1])
#         )

#         p2 = (
#             int(points[i][0]),
#             int(points[i][1])
#         )

#         cv2.line(
#             frame,
#             p1,
#             p2,
#             (0, 0, 255),
#             2
#         )


# # ============================================================
# # Main
# # ============================================================

# def main():

#     print("=" * 60)
#     print("Trajectory Analyzer")
#     print("=" * 60)

#     # --------------------------------------------------------
#     # Load model
#     # --------------------------------------------------------

#     print("\nLoading model...")

#     model = YOLO(
#         MODEL_PATH
#     )

#     print("Model loaded.")

#     # --------------------------------------------------------
#     # Load ROIs
#     # --------------------------------------------------------

#     print("\nLoading ROIs...")

#     rois = load_rois(
#         ROI_CONFIG
#     )

#     print(
#         f"Loaded {len(rois)} ROIs:"
#     )

#     for name in rois:
#         print(f"  - {name}")

#     # --------------------------------------------------------
#     # Open video
#     # --------------------------------------------------------

#     print("\nOpening video...")

#     cap = cv2.VideoCapture(
#         VIDEO_PATH
#     )

#     if not cap.isOpened():

#         raise RuntimeError(
#             f"Could not open video: {VIDEO_PATH}"
#         )

#     fps = cap.get(
#         cv2.CAP_PROP_FPS
#     )

#     width = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_WIDTH
#         )
#     )

#     height = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_HEIGHT
#         )
#     )

#     total_frames = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_COUNT
#         )
#     )

#     print(
#         f"Resolution: {width} x {height}"
#     )

#     print(
#         f"FPS: {fps}"
#     )

#     print(
#         f"Total frames: {total_frames}"
#     )

#     print(
#         f"Processing frames: "
#         f"{min(MAX_FRAMES, total_frames)}"
#     )

#     # --------------------------------------------------------
#     # Video writer
#     # --------------------------------------------------------

#     fourcc = cv2.VideoWriter_fourcc(
#         *"mp4v"
#     )

#     writer = cv2.VideoWriter(
#         OUTPUT_PATH,
#         fourcc,
#         fps,
#         (width, height)
#     )

#     if not writer.isOpened():

#         raise RuntimeError(
#             f"Could not create output: "
#             f"{OUTPUT_PATH}"
#         )

#     # ========================================================
#     # Per-person state
#     # ========================================================

#     # Smoothed trajectory history
#     trajectories = defaultdict(
#         lambda: deque(
#             maxlen=TRAIL_LENGTH
#         )
#     )

#     # Current smoothed position
#     smoothed_positions = {}

#     # Smoothed speed
#     smoothed_speed = {}

#     # Current ROI
#     current_rois = {}

#     # ========================================================
#     # Frame processing
#     # ========================================================

#     frame_number = 0

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_number += 1

#         if (
#             MAX_FRAMES is not None
#             and frame_number > MAX_FRAMES
#         ):
#             break

#         # ----------------------------------------------------
#         # Detection + Tracking
#         # ----------------------------------------------------

#         results = model.track(
#             frame,
#             persist=True,
#             classes=[0],
#             conf=CONFIDENCE,
#             tracker=TRACKER_CONFIG,
#             verbose=False,
#             device=0
#         )

#         output_frame = frame.copy()

#         # ----------------------------------------------------
#         # Draw ROIs
#         # ----------------------------------------------------

#         draw_rois(
#             output_frame,
#             rois
#         )

#         result = results[0]

#         # ----------------------------------------------------
#         # Check tracked detections
#         # ----------------------------------------------------

#         if (
#             result.boxes is not None
#             and result.boxes.id is not None
#         ):

#             boxes = (
#                 result.boxes.xyxy
#                 .cpu()
#                 .numpy()
#             )

#             track_ids = (
#                 result.boxes.id
#                 .cpu()
#                 .numpy()
#                 .astype(int)
#             )

#             # ------------------------------------------------
#             # Process every tracked person
#             # ------------------------------------------------

#             for box, track_id in zip(
#                 boxes,
#                 track_ids
#             ):

#                 # =================================================
#                 # 1. Raw position
#                 # =================================================

#                 raw_position = bottom_center(
#                     box
#                 )

#                 raw_position_array = np.array(
#                     raw_position,
#                     dtype=np.float32
#                 )

#                 # =================================================
#                 # 2. Smooth position
#                 # =================================================

#                 if track_id not in smoothed_positions:

#                     smoothed_positions[
#                         track_id
#                     ] = raw_position_array

#                 else:

#                     previous_position = (
#                         smoothed_positions[
#                             track_id
#                         ]
#                     )

#                     smoothed_positions[
#                         track_id
#                     ] = (
#                         POSITION_SMOOTHING
#                         * raw_position_array
#                         +
#                         (
#                             1.0
#                             - POSITION_SMOOTHING
#                         )
#                         * previous_position
#                     )

#                 smoothed_position = (
#                     smoothed_positions[
#                         track_id
#                     ]
#                 )

#                 smoothed_point = (
#                     int(smoothed_position[0]),
#                     int(smoothed_position[1])
#                 )

#                 # =================================================
#                 # 3. Store smoothed trajectory
#                 # =================================================

#                 trajectories[
#                     track_id
#                 ].append(
#                     smoothed_point
#                 )

#                 history = trajectories[
#                     track_id
#                 ]

#                 # =================================================
#                 # 4. Calculate motion
#                 # =================================================

#                 raw_speed, direction = (
#                     calculate_motion(
#                         history,
#                         fps
#                     )
#                 )

#                 # =================================================
#                 # 5. Smooth speed
#                 # =================================================

#                 if track_id not in smoothed_speed:

#                     smoothed_speed[
#                         track_id
#                     ] = raw_speed

#                 else:

#                     smoothed_speed[
#                         track_id
#                     ] = (
#                         SPEED_SMOOTHING
#                         * raw_speed
#                         +
#                         (
#                             1.0
#                             - SPEED_SMOOTHING
#                         )
#                         * smoothed_speed[
#                             track_id
#                         ]
#                     )

#                 speed = smoothed_speed[
#                     track_id
#                 ]

#                 # =================================================
#                 # 6. ROI
#                 #
#                 # Use RAW position for ROI membership.
#                 # Do not use smoothed position here.
#                 # =================================================

#                 roi = get_roi(
#                     raw_position,
#                     rois
#                 )

#                 current_rois[
#                     track_id
#                 ] = roi

#                 # =================================================
#                 # 7. Bounding box
#                 # =================================================

#                 x1, y1, x2, y2 = (
#                     map(
#                         int,
#                         box
#                     )
#                 )

#                 cv2.rectangle(
#                     output_frame,
#                     (x1, y1),
#                     (x2, y2),
#                     (0, 255, 0),
#                     2
#                 )

#                 # =================================================
#                 # 8. Draw trajectory
#                 # =================================================

#                 draw_trajectory(
#                     output_frame,
#                     history
#                 )

#                 # =================================================
#                 # 9. Draw direction
#                 # =================================================

#                 draw_direction_arrow(
#                     output_frame,
#                     smoothed_point,
#                     direction
#                 )

#                 # =================================================
#                 # 10. Draw ID
#                 # =================================================

#                 cv2.putText(
#                     output_frame,
#                     f"ID {track_id}",
#                     (
#                         x1,
#                         max(
#                             20,
#                             y1 - 10
#                         )
#                     ),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2
#                 )

#                 # =================================================
#                 # 11. Draw speed
#                 # =================================================

#                 cv2.putText(
#                     output_frame,
#                     f"Speed: {speed:.1f} px/s",
#                     (
#                         x1,
#                         y2 + 20
#                     ),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 255),
#                     2
#                 )

#                 # =================================================
#                 # 12. Draw ROI
#                 # =================================================

#                 if roi is not None:

#                     cv2.putText(
#                         output_frame,
#                         roi,
#                         (
#                             x1,
#                             y2 + 40
#                         ),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (255, 255, 255),
#                         2
#                     )

#         # --------------------------------------------------------
#         # Frame information
#         # --------------------------------------------------------

#         cv2.putText(
#             output_frame,
#             f"Frame: {frame_number}",
#             (20, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.putText(
#             output_frame,
#             f"Active tracks: {len(track_ids) if result.boxes is not None and result.boxes.id is not None else 0}",
#             (20, 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         # --------------------------------------------------------
#         # Write frame
#         # --------------------------------------------------------

#         writer.write(
#             output_frame
#         )

#         # --------------------------------------------------------
#         # Progress
#         # --------------------------------------------------------

#         if frame_number % 100 == 0:

#             print(
#                 f"Processed "
#                 f"{frame_number}/"
#                 f"{min(MAX_FRAMES, total_frames)}"
#             )

#     # ========================================================
#     # Cleanup
#     # ========================================================

#     cap.release()
#     writer.release()

#     print("\n" + "=" * 60)
#     print("Trajectory analysis completed.")
#     print("=" * 60)

#     print(
#         f"Output: {OUTPUT_PATH}"
#     )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
# from collections import defaultdict, deque
# from ultralytics import YOLO
# import yaml


# # ---------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------

# VIDEO_PATH = "data/entrance.mp4"
# OUTPUT_PATH = "outputs/trajectory_analysis.mp4"

# MODEL_PATH = "yolo26m.pt"
# TRACKER_CONFIG = "botsort.yaml"

# CONFIDENCE = 0.4
# MAX_FRAMES = 6000

# # Number of trajectory points retained per person
# TRAIL_LENGTH = 300

# # Smoothing factor for speed estimation
# SPEED_SMOOTHING = 0.3


# # ---------------------------------------------------------
# # ROI utilities
# # ---------------------------------------------------------

# def load_rois(path):
#     with open(path, "r") as f:
#         data = yaml.safe_load(f)

#     return {
#         name: np.array(roi["polygon"], dtype=np.int32)
#         for name, roi in data["rois"].items()
#     }


# def get_roi(point, rois):
#     """
#     Return the ROI containing the point.

#     If the point is not inside any ROI, return None.
#     """
#     x, y = point

#     for name, polygon in rois.items():
#         if cv2.pointPolygonTest(
#             polygon,
#             (float(x), float(y)),
#             False
#         ) >= 0:
#             return name

#     return None


# # ---------------------------------------------------------
# # Trajectory utilities
# # ---------------------------------------------------------

# def bottom_center(box):
#     """
#     Calculate the bottom-center point of a bounding box.

#     This approximates the person's position on the ground.
#     """
#     x1, y1, x2, y2 = box

#     x = int((x1 + x2) / 2)
#     y = int(y2)

#     return x, y


# def calculate_motion(history, fps):
#     """
#     Estimate velocity, speed and direction from trajectory history.

#     Returns:
#         speed_px_s
#         direction_deg
#     """

#     if len(history) < 2:
#         return 0.0, None

#     p1 = np.array(history[-2], dtype=float)
#     p2 = np.array(history[-1], dtype=float)

#     displacement = p2 - p1

#     distance = np.linalg.norm(displacement)

#     speed_px_s = distance * fps

#     dx = displacement[0]
#     dy = displacement[1]

#     direction_deg = np.degrees(
#         np.arctan2(dy, dx)
#     )

#     return speed_px_s, direction_deg


# # ---------------------------------------------------------
# # Drawing
# # ---------------------------------------------------------

# def draw_rois(frame, rois):
#     for name, polygon in rois.items():

#         cv2.polylines(
#             frame,
#             [polygon],
#             isClosed=True,
#             color=(255, 255, 0),
#             thickness=2,
#         )

#         # Label near first polygon point
#         x, y = polygon[0]

#         cv2.putText(
#             frame,
#             name,
#             (int(x), int(y) - 8),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (255, 255, 0),
#             2,
#         )


# def draw_arrow(frame, position, direction_deg, length=35):

#     if direction_deg is None:
#         return

#     angle = np.radians(direction_deg)

#     dx = int(np.cos(angle) * length)
#     dy = int(np.sin(angle) * length)

#     start = tuple(map(int, position))

#     end = (
#         int(position[0] + dx),
#         int(position[1] + dy),
#     )

#     cv2.arrowedLine(
#         frame,
#         start,
#         end,
#         (0, 255, 255),
#         2,
#         tipLength=0.25,
#     )


# # ---------------------------------------------------------
# # Main
# # ---------------------------------------------------------

# def main():

#     print("Loading model...")

#     model = YOLO(MODEL_PATH)

#     print("Loading ROIs...")

#     rois = load_rois("configs/entrance.yaml")

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     if not cap.isOpened():
#         raise RuntimeError(
#             f"Could not open video: {VIDEO_PATH}"
#         )

#     fps = cap.get(cv2.CAP_PROP_FPS)

#     width = int(
#         cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     )

#     height = int(
#         cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     )

#     print(f"Video: {width}x{height}")
#     print(f"FPS: {fps}")

#     fourcc = cv2.VideoWriter_fourcc(
#         *"mp4v"
#     )

#     writer = cv2.VideoWriter(
#         OUTPUT_PATH,
#         fourcc,
#         fps,
#         (width, height),
#     )

#     # -----------------------------------------------------
#     # Per-person trajectory history
#     # -----------------------------------------------------

#     trajectories = defaultdict(
#         lambda: deque(maxlen=TRAIL_LENGTH)
#     )

#     # Smoothed speed for each person
#     smoothed_speed = {}

#     # Previous ROI for each person
#     current_rois = {}

#     frame_number = 0

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_number += 1

#         if MAX_FRAMES and frame_number > MAX_FRAMES:
#             break

#         # -------------------------------------------------
#         # Detection + tracking
#         # -------------------------------------------------

#         results = model.track(
#             frame,
#             persist=True,
#             classes=[0],
#             conf=CONFIDENCE,
#             tracker=TRACKER_CONFIG,
#             verbose=False,
#             device=0,
#         )

#         annotated = frame.copy()

#         draw_rois(
#             annotated,
#             rois
#         )

#         result = results[0]

#         if (
#             result.boxes is not None
#             and result.boxes.id is not None
#         ):

#             boxes = result.boxes.xyxy.cpu().numpy()
#             track_ids = (
#                 result.boxes.id
#                 .cpu()
#                 .numpy()
#                 .astype(int)
#             )

#             # -------------------------------------------------
#             # Process every tracked person
#             # -------------------------------------------------

#             for box, track_id in zip(
#                 boxes,
#                 track_ids
#             ):

#                 position = bottom_center(box)

#                 # Add trajectory point
#                 trajectories[track_id].append(
#                     position
#                 )

#                 history = trajectories[track_id]

#                 # Calculate motion
#                 speed, direction = calculate_motion(
#                     history,
#                     fps
#                 )

#                 # Smooth speed
#                 if track_id not in smoothed_speed:

#                     smoothed_speed[track_id] = speed

#                 else:

#                     smoothed_speed[track_id] = (
#                         SPEED_SMOOTHING * speed
#                         +
#                         (1 - SPEED_SMOOTHING)
#                         * smoothed_speed[track_id]
#                     )

#                 speed = smoothed_speed[track_id]

#                 # Determine current ROI
#                 roi = get_roi(
#                     position,
#                     rois
#                 )

#                 current_rois[track_id] = roi

#                 # -------------------------------------------------
#                 # Draw bounding box
#                 # -------------------------------------------------

#                 x1, y1, x2, y2 = (
#                     map(int, box)
#                 )

#                 cv2.rectangle(
#                     annotated,
#                     (x1, y1),
#                     (x2, y2),
#                     (0, 255, 0),
#                     2,
#                 )

#                 # -------------------------------------------------
#                 # Draw trajectory
#                 # -------------------------------------------------

#                 points = list(history)

#                 for i in range(1, len(points)):

#                     cv2.line(
#                         annotated,
#                         points[i - 1],
#                         points[i],
#                         (0, 0, 255),
#                         2,
#                     )

#                 # -------------------------------------------------
#                 # Draw direction arrow
#                 # -------------------------------------------------

#                 draw_arrow(
#                     annotated,
#                     position,
#                     direction
#                 )

#                 # -------------------------------------------------
#                 # Draw ID
#                 # -------------------------------------------------

#                 label = f"ID {track_id}"

#                 cv2.putText(
#                     annotated,
#                     label,
#                     (x1, max(20, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2,
#                 )

#                 # -------------------------------------------------
#                 # Draw motion information
#                 # -------------------------------------------------

#                 motion_text = (
#                     f"{speed:.1f}px/s"
#                 )

#                 cv2.putText(
#                     annotated,
#                     motion_text,
#                     (x1, y2 + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 255),
#                     2,
#                 )

#                 # -------------------------------------------------
#                 # Draw ROI
#                 # -------------------------------------------------

#                 if roi is not None:

#                     cv2.putText(
#                         annotated,
#                         roi,
#                         (x1, y2 + 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (255, 255, 255),
#                         2,
#                     )

#         # -----------------------------------------------------
#         # Global information
#         # -----------------------------------------------------

#         cv2.putText(
#             annotated,
#             f"Frame: {frame_number}",
#             (20, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#         )

#         cv2.putText(
#             annotated,
#             f"Tracked persons: {len(trajectories)}",
#             (20, 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#         )

#         writer.write(annotated)

#         if frame_number % 100 == 0:

#             print(
#                 f"Processed {frame_number} frames"
#             )

#     cap.release()
#     writer.release()

#     print()
#     print("Trajectory analysis completed.")
#     print(f"Output: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()


