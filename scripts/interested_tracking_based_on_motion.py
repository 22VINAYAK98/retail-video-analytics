
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
OUTPUT_PATH = "outputs/behavioral_interest_analysis.mp4"

MODEL_PATH = "yolo26m.pt"
TRACKER_CONFIG = "botsort.yaml"
ROI_CONFIG = "configs/entrance.yaml"

DETAIL_CSV = "outputs/behavioral_interest_tracks.csv"
FRAME_CSV = "outputs/behavioral_interest_frame_metrics.csv"
SUMMARY_CSV = "outputs/behavioral_interest_summary.csv"
PLOT_DIR = "outputs/behavioral_interest_plots"

CONFIDENCE = 0.40
DEVICE = 0
MAX_FRAMES = 6000

# RLS trajectory.
TRAIL_LENGTH = 300

# Local least-squares speed trend.
SPEED_TREND_WINDOW_SECONDS = 0.75
MIN_TREND_SAMPLES = 8

# Personal baseline.
BASELINE_WINDOW_SECONDS = 1.0
BASELINE_MIN_SAMPLES = 8
BASELINE_MIN_SPEED = 20.0

# Interest from motion.
SLOWDOWN_RATIO = 0.70
SLOWDOWN_SLOPE = -3.0
MIN_SLOWDOWN_DURATION = 0.50

STOP_RATIO = 0.25
MIN_STOP_DURATION = 0.75

# Interest is detected BEFORE entry, while the person is still OUT,
# using distance to the configured ENTERING ROI. This is not a new ROI.
INTEREST_PROXIMITY_PIXELS = 150.0

# Initial trajectory classification.
# We intentionally do NOT classify from one noisy detection.
INITIAL_STATE_SAMPLES = 8

# ROI name matching. These refer ONLY to names already present
# in configs/entrance.yaml. No FRONT ROI is invented.
ENTERING_KEYWORDS = ("entering", "entry", "entrance")
IN_KEYWORDS = ("inside", "in_store", "store_inside", "interior", "in")

# A track is considered plotted if sufficiently long.
MIN_TRACK_SAMPLES_FOR_PLOT = 20


# ============================================================
# ROI helpers
# ============================================================

def load_rois(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    rois = {}
    for name, roi in data["rois"].items():
        rois[name] = np.asarray(roi["polygon"], dtype=np.int32)
    return rois


def resolve_roi_roles(rois):
    """
    Resolve semantic roles from the EXISTING ROI names.

    We use only configured ROIs:
        ENTERING = ROI name contains entering/entry/entrance
        IN       = ROI name contains inside/in_store/store_inside/interior

    Everything else remains an ordinary configured ROI and is NOT invented
    as a FRONT state. A point outside the configured IN/ENTERING ROIs is OUT
    for the entry state machine.
    """
    entering = []
    inside = []

    for name in rois:
        lname = name.lower().replace("-", "_").replace(" ", "_")

        # An explicitly entering ROI must never also be treated as IN.
        # This is important for names such as "inside_store_entering".
        is_entering = any(k in lname for k in ENTERING_KEYWORDS)

        if is_entering:
            entering.append(name)
            continue

        # "in" is intentionally matched conservatively to avoid names
        # such as "interest" being interpreted as IN.
        if (
            any(k in lname for k in IN_KEYWORDS if k != "in")
            or lname == "in"
            or lname.startswith("in_")
            or lname.endswith("_in")
        ):
            inside.append(name)

    return entering, inside


def point_inside(point, polygon):
    if polygon is None:
        return False

    return cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        False
    ) >= 0


def point_in_any_roi(point, rois, names):
    for name in names:
        if name in rois and point_inside(point, rois[name]):
            return name
    return None


def distance_to_any_roi(point, rois, names):
    """Minimum geometric distance from point to configured ROI polygons."""
    distances = []
    for name in names:
        if name not in rois:
            continue
        d = cv2.pointPolygonTest(
            rois[name],
            (float(point[0]), float(point[1])),
            True
        )
        # Signed distance: positive inside, negative outside.
        distances.append(float(d))
    if not distances:
        return float("inf"), None
    # For proximity, use absolute distance to the nearest boundary.
    idx = int(np.argmin(np.abs(distances)))
    return abs(distances[idx]), list(names)[idx]


def classify_region(point, rois, entering_names, inside_names):
    """
    IMPORTANT:
    Region is derived ONLY from the configured ROI polygons.

    Priority:
        IN > ENTERING > OUT

    OUT means the trajectory point is outside both configured
    ENTERING and IN ROIs.
    """
    inside_roi = point_in_any_roi(point, rois, inside_names)
    if inside_roi is not None:
        return "IN", inside_roi

    entering_roi = point_in_any_roi(point, rois, entering_names)
    if entering_roi is not None:
        return "ENTERING", entering_roi

    return "OUT", None


# ============================================================
# RLS
# ============================================================

class RecursiveLeastSquares:
    """Online RLS model y = theta0 + theta1*t."""

    def __init__(self, forgetting_factor=0.985, initial_covariance=1000.0):
        self.lam = forgetting_factor
        self.theta = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * initial_covariance
        self.initialized = False
        self.t0 = None

    def update(self, t, value):
        if self.t0 is None:
            self.t0 = float(t)

        tau = float(t) - self.t0
        phi = np.array([[1.0], [tau]], dtype=np.float64)

        Pphi = self.P @ phi
        denom = self.lam + float(phi.T @ Pphi)
        K = Pphi / max(denom, 1e-12)

        prediction = float(phi.T @ self.theta)
        error = float(value) - prediction

        self.theta = self.theta + K * error
        self.P = (self.P - K @ phi.T @ self.P) / self.lam
        self.initialized = True

        return self.predict(t), float(self.theta[1, 0])

    def predict(self, t):
        if not self.initialized:
            return 0.0

        tau = float(t) - self.t0
        return float(self.theta[0, 0] + self.theta[1, 0] * tau)


def linear_least_squares(times, values):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(times) < 2:
        return 0.0, 0.0

    centered = times - times[-1]
    A = np.column_stack((np.ones(len(centered)), centered))

    try:
        coeff, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        return float(coeff[0]), float(coeff[1])
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def local_speed_trend(speed_history, current_time):
    selected = [
        (t, s)
        for t, s in speed_history
        if current_time - t <= SPEED_TREND_WINDOW_SECONDS
    ]

    if len(selected) < MIN_TREND_SAMPLES:
        return 0.0

    t = np.asarray([x[0] for x in selected])
    s = np.asarray([x[1] for x in selected])

    _, slope = linear_least_squares(t, s)
    return float(slope)


# ============================================================
# Per-person state
# ============================================================

class PersonState:

    def __init__(self, track_id):
        self.track_id = int(track_id)

        # (time, x, y)
        self.positions = deque(maxlen=TRAIL_LENGTH)

        # (time, RLS speed)
        self.speeds = deque(maxlen=TRAIL_LENGTH)

        # (time, semantic region, ROI name)
        self.region_history = []  # ALL ROI observations; never truncated
        self.initial_region_observed = None

        self.metrics = []

        self.rls_x = RecursiveLeastSquares()
        self.rls_y = RecursiveLeastSquares()
        self.rls_v = RecursiveLeastSquares()

        self.baseline_speed = None
        self.baseline_samples = []
        self.baseline_locked = False

        self.slowdown_start = None
        self.stop_start = None

        self.max_speed_reduction = 0.0
        self.max_slowdown_duration = 0.0
        self.max_stop_duration = 0.0

        # Final behavioral state.
        self.interested = False
        self.entered = False

        self.entry_time = None
        self.entry_roi = None

        self.interest_reasons = set()

        # Initial trajectory state.
        self.initial_state = None
        self.initial_roi = None
        self.initial_state_decided = False
        self.ignore_interest = False

        # Entry state machine.
        self.entry_detected_from = None

        self.last_raw_position = None
        self.last_seen_time = None

    def update_rls(self, t, x, y, raw_speed):
        sx, vx = self.rls_x.update(t, x)
        sy, vy = self.rls_y.update(t, y)

        position_speed = float(np.hypot(vx, vy))

        self.rls_v.update(t, raw_speed)
        scalar_rls_speed = self.rls_v.predict(t)

        return sx, sy, vx, vy, position_speed, scalar_rls_speed


# ============================================================
# Trajectory entry logic
# ============================================================

def decide_initial_state(state):
    """
    Freeze eligibility from the FIRST trajectory ROI observation.

    This is deliberately not a majority vote over later frames.
    The first observed position is the person's origin for this video track:

        first ROI = IN       -> already inside -> ignore forever
        first ROI = ENTERING -> interested + entered
        first ROI = OUT      -> valid candidate; entry can only happen later

    This is what prevents an IN -> ENTERING -> OUT person from being counted
    as a new entrant just because the tracker later observes them outside.
    """
    if state.initial_state_decided:
        return

    if not state.region_history:
        return

    # The first stored observation is the only observation used to establish
    # origin.  We never replace this decision later.
    first_time, first_region, first_roi = state.region_history[0]
    state.initial_region_observed = first_region
    state.initial_state = first_region
    state.initial_roi = first_roi
    state.initial_state_decided = True

    if first_region == "IN":
        state.ignore_interest = True
        state.interest_reasons.add("initially_inside")

    elif first_region == "ENTERING":
        # User requirement: a track that starts in ENTERING is already an
        # interested entrant.
        state.interested = True
        state.entered = True
        state.entry_time = first_time
        state.entry_roi = first_roi
        state.entry_detected_from = "initial_trajectory"
        state.interest_reasons.add("initially_entering")


def detect_entry_from_trajectory(state):
    """
    Detect entry ONLY for tracks whose origin was OUT.

    The decision is based on the complete RLS trajectory history. We accept:
        OUT -> ENTERING
        OUT -> IN
        OUT -> ENTERING -> IN

    We do NOT accept an entry for an initially-IN track, so an
    IN -> ENTERING -> OUT or IN -> OUT trajectory can never become an entrant.
    """
    if state.ignore_interest or state.entered:
        return None
    if state.initial_state != "OUT":
        return None

    history = state.region_history
    if len(history) < 2:
        return None

    # Initial state is fixed. Any later ENTERING/IN is a valid entry event.
    for timestamp, region, roi_name in history[1:]:
        if region in ("ENTERING", "IN"):
            return timestamp, region, roi_name

    return None


def update_entry_state(state):
    """Commit a permanent entry event from the trajectory ROI history."""
    if state.ignore_interest or state.entered:
        return

    evidence = detect_entry_from_trajectory(state)
    if evidence is None:
        return

    timestamp, region, roi_name = evidence
    state.entered = True
    state.entry_time = timestamp
    state.entry_roi = roi_name or region
    state.entry_detected_from = "trajectory_history"
    state.interest_reasons.add("entered_store")
    state.interested = True


# ============================================================
# Drawing
# ============================================================

def draw_rois(frame, rois, entering_names, inside_names):
    """
    Draw ONLY configured ROIs.

    No FRONT ROI/state is created.
    """
    ENTERING_COLOR = (0, 200, 0)
    IN_COLOR = (255, 0, 0)
    OTHER_COLOR = (0, 220, 220)

    for name, polygon in rois.items():

        if name in inside_names:
            color = IN_COLOR
            tag = "[IN]"
        elif name in entering_names:
            color = ENTERING_COLOR
            tag = "[ENTERING]"
        else:
            color = OTHER_COLOR
            tag = "[ROI]"

        cv2.polylines(
            frame,
            [polygon],
            True,
            color,
            3
        )

        x, y = polygon[0]

        cv2.putText(
            frame,
            f"{name} {tag}",
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


def draw_trajectory(frame, trajectory):
    pts = list(trajectory)

    for i in range(1, len(pts)):
        p1 = tuple(map(int, pts[i - 1]))
        p2 = tuple(map(int, pts[i]))

        cv2.line(
            frame,
            p1,
            p2,
            (0, 0, 255),
            2
        )


def draw_arrow(frame, position, direction, length=40):
    if direction is None:
        return

    angle = np.radians(direction)

    start = tuple(map(int, position))

    end = (
        int(position[0] + np.cos(angle) * length),
        int(position[1] + np.sin(angle) * length)
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
# Plots
# ============================================================

def generate_track_plot(track_id, metrics, state, output_dir):

    if len(metrics) < MIN_TRACK_SAMPLES_FOR_PLOT:
        return

    os.makedirs(output_dir, exist_ok=True)

    t = np.asarray([m["time"] for m in metrics])
    raw_x = np.asarray([m["raw_x"] for m in metrics])
    raw_y = np.asarray([m["raw_y"] for m in metrics])
    sx = np.asarray([m["smooth_x"] for m in metrics])
    sy = np.asarray([m["smooth_y"] for m in metrics])

    speed = np.asarray([m["speed"] for m in metrics])
    raw_speed = np.asarray([m["raw_speed"] for m in metrics])
    trend = np.asarray([m["speed_trend"] for m in metrics])
    ratio = np.asarray([m["speed_ratio"] for m in metrics])

    interested = np.asarray(
        [m["interested"] for m in metrics]
    ).astype(bool)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 15)
    )

    axes[0].plot(
        raw_x,
        raw_y,
        label="Raw trajectory",
        alpha=0.45
    )

    axes[0].plot(
        sx,
        sy,
        label="RLS trajectory",
        linewidth=2
    )

    axes[0].scatter(
        sx[0],
        sy[0],
        s=40,
        label="Start"
    )

    axes[0].scatter(
        sx[-1],
        sy[-1],
        s=40,
        label="End"
    )

    axes[0].invert_yaxis()
    axes[0].set_title(
        f"Track {track_id} - Trajectory"
    )
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t,
        raw_speed,
        label="Raw speed",
        alpha=0.35
    )

    axes[1].plot(
        t,
        speed,
        label="RLS trajectory speed",
        linewidth=2
    )

    axes[1].set_title(
        f"Track {track_id} - Speed"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (px/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        t,
        trend,
        label="Local speed trend"
    )

    axes[2].axhline(
        0,
        linewidth=1
    )

    axes[2].set_title(
        f"Track {track_id} - Speed Trend"
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("dv/dt (px/s²)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        t,
        ratio,
        label="Speed / personal baseline"
    )

    axes[3].axhline(
        SLOWDOWN_RATIO,
        linestyle="--",
        label="Slowdown ratio"
    )

    axes[3].axhline(
        STOP_RATIO,
        linestyle=":",
        label="Stop ratio"
    )

    if np.any(interested):
        axes[3].fill_between(
            t,
            0,
            1,
            where=interested,
            alpha=0.2,
            label="Interested"
        )

    axes[3].set_ylim(bottom=0)

    axes[3].set_title(
        f"Track {track_id} - Relative Behavior | "
        f"{', '.join(sorted(state.interest_reasons)) or 'no interest evidence'}"
    )

    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Speed ratio")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(
        f"Behavioral Motion Analysis - ID {track_id} | "
        f"Initial={state.initial_state} | "
        f"Interested={state.interested} | "
        f"Entered={state.entered}",
        fontsize=15
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            f"ID_{track_id}.png"
        ),
        dpi=150
    )

    plt.close(fig)


# ============================================================
# CSV
# ============================================================

def write_frame_csv(all_metrics, path):

    fields = [
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
        "speed_trend_px_s2",
        "baseline_speed_px_s",
        "speed_ratio",
        "region_status",
        "region_roi",
        "initial_state",
        "initial_roi",
        "entered",
        "entry_roi",
        "interested",
        "ignored_initially_inside",
        "interest_reasons",
    ]

    with open(path, "w", newline="") as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields
        )

        w.writeheader()

        for tid, rows in all_metrics.items():

            for m in rows:

                w.writerow({
                    "track_id": tid,
                    "frame": m["frame"],
                    "time_sec": m["time"],
                    "raw_x": m["raw_x"],
                    "raw_y": m["raw_y"],
                    "smooth_x": m["smooth_x"],
                    "smooth_y": m["smooth_y"],
                    "vx_px_s": m["vx"],
                    "vy_px_s": m["vy"],
                    "raw_speed_px_s": m["raw_speed"],
                    "speed_px_s": m["speed"],
                    "speed_trend_px_s2": m["speed_trend"],
                    "baseline_speed_px_s": m["baseline_speed"],
                    "speed_ratio": m["speed_ratio"],
                    "region_status": m["region_status"],
                    "region_roi": m["region_roi"] or "",
                    "initial_state": m["initial_state"] or "",
                    "initial_roi": m.get("initial_roi") or "",
                    "entered": int(m["entered"]),
                    "entry_roi": m["entry_roi"] or "",
                    "interested": int(m["interested"]),
                    "ignored_initially_inside": int(
                        m["ignored_initially_inside"]
                    ),
                    "interest_reasons": "+".join(
                        m["interest_reasons"]
                    ),
                })


def compress_region_sequence(history):
    """Compress consecutive identical ROI states for human-auditable CSV output."""
    seq = []
    last = None
    for _, region, _ in history:
        if region != last:
            seq.append(region)
            last = region
    return seq


def write_track_csv(states, path):

    fields = [
        "track_id",
        "initial_state",
        "initial_roi",
        "interested",
        "entered",
        "ignored_initially_inside",
        "classification",
        "interest_reasons",
        "baseline_speed_px_s",
        "minimum_speed_px_s",
        "maximum_speed_reduction_fraction",
        "max_slowdown_duration_sec",
        "max_stop_duration_sec",
        "entry_time_sec",
        "entry_roi",
        "entry_detected_from",
        "roi_trajectory",
    ]

    with open(path, "w", newline="") as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields
        )

        w.writeheader()

        for tid in sorted(states):

            s = states[tid]

            speeds = [
                m["speed"]
                for m in s.metrics
            ]

            min_speed = (
                min(speeds)
                if speeds
                else 0.0
            )

            if s.ignore_interest:
                classification = "ignored_initially_inside"

            elif s.interested and s.entered:
                classification = "interested_entered"

            elif s.interested:
                classification = "interested_passed"

            elif s.entered:
                classification = "entered_directly"

            else:
                classification = "passed_by"

            w.writerow({
                "track_id": tid,
                "initial_state": s.initial_state or "",
                "initial_roi": s.initial_roi or "",
                "interested": int(s.interested),
                "entered": int(s.entered),
                "ignored_initially_inside": int(
                    s.ignore_interest
                ),
                "classification": classification,
                "interest_reasons": "+".join(
                    sorted(s.interest_reasons)
                ),
                "baseline_speed_px_s": (
                    ""
                    if s.baseline_speed is None
                    else round(s.baseline_speed, 3)
                ),
                "minimum_speed_px_s": round(
                    min_speed,
                    3
                ),
                "maximum_speed_reduction_fraction": round(
                    s.max_speed_reduction,
                    4
                ),
                "max_slowdown_duration_sec": round(
                    s.max_slowdown_duration,
                    3
                ),
                "max_stop_duration_sec": round(
                    s.max_stop_duration,
                    3
                ),
                "entry_time_sec": (
                    ""
                    if s.entry_time is None
                    else round(s.entry_time, 3)
                ),
                "entry_roi": s.entry_roi or "",
                "entry_detected_from": (
                    s.entry_detected_from or ""
                ),
                "roi_trajectory": " -> ".join(
                    [r for r in compress_region_sequence(s.region_history)]
                ),
            })


def write_summary(states, path):

    eligible = [
        s
        for s in states.values()
        if not s.ignore_interest
    ]

    total = len(states)

    interested = sum(
        s.interested
        for s in eligible
    )

    interested_entered = sum(
        s.interested and s.entered
        for s in eligible
    )

    interested_passed = sum(
        s.interested and not s.entered
        for s in eligible
    )

    not_interested = sum(
        not s.interested
        for s in eligible
    )

    rows = [
        ("total_unique_people", total),
        (
            "ignored_initially_inside",
            sum(
                s.ignore_interest
                for s in states.values()
            )
        ),
        (
            "initially_entering",
            sum(
                s.initial_state == "ENTERING"
                for s in states.values()
            )
        ),
        ("eligible_people", len(eligible)),
        ("interested", interested),
        ("interested_entered", interested_entered),
        ("interested_passed", interested_passed),
        ("not_interested", not_interested),
        (
            "interest_reason_sustained_slowdown",
            sum(
                "sustained_slowdown"
                in s.interest_reasons
                for s in states.values()
            )
        ),
        (
            "interest_reason_near_stop",
            sum(
                "near_stop"
                in s.interest_reasons
                for s in states.values()
            )
        ),
        (
            "interest_reason_entered_store",
            sum(
                "entered_store"
                in s.interest_reasons
                for s in states.values()
            )
        ),
    ]

    with open(path, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow([
            "metric",
            "count"
        ])

        w.writerows(rows)


def generate_summary_pie_chart(states, output_path):

    eligible = [
        s
        for s in states.values()
        if not s.ignore_interest
    ]

    interested_entered = sum(
        s.interested and s.entered
        for s in eligible
    )

    interested_passed = sum(
        s.interested and not s.entered
        for s in eligible
    )

    not_interested = sum(
        not s.interested
        for s in eligible
    )

    interested = (
        interested_entered
        + interested_passed
    )

    labels = [
        "Interested + Entered",
        "Interested + Passed",
        "Not Interested",
    ]

    values = [
        interested_entered,
        interested_passed,
        not_interested,
    ]

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    if sum(values) > 0:

        ax.pie(
            values,
            labels=labels,
            autopct=lambda p:
                f"{p:.1f}%\n"
                f"({int(round(p * sum(values) / 100.0))})",
            startangle=90,
            wedgeprops={"width": 0.42},
        )

        ax.text(
            0,
            0,
            f"Interested\n{interested}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No eligible people detected",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes
        )

    ax.set_title(
        "Interest Outcomes"
    )

    ax.axis("equal")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("RLS Behavioral Interest / Entered / Passed Analysis")
    print("=" * 72)

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    print("\nLoading model...")

    model = YOLO(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print("\nLoading ROI configuration...")

    rois = load_rois(
        ROI_CONFIG
    )

    entering_names, inside_names = resolve_roi_roles(
        rois
    )

    print("\nConfigured ROIs:")

    for name in rois:
        print(f"  {name}")

    print(
        f"\nENTERING ROIs: {entering_names}"
    )

    print(
        f"IN ROIs:       {inside_names}"
    )

    if not entering_names and not inside_names:
        raise RuntimeError(
            "Could not resolve ENTERING/IN ROIs from "
            f"{ROI_CONFIG}. Check the actual ROI names."
        )

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
        f"Frames: {frames_to_process}/{total_frames}"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    states = {}
    all_metrics = defaultdict(list)

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if (
            not ret
            or frame_number >= frames_to_process
        ):
            break

        frame_number += 1

        current_time = (
            frame_number - 1
        ) / fps

        output = frame.copy()

        draw_rois(
            output,
            rois,
            entering_names,
            inside_names
        )

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            tracker=TRACKER_CONFIG,
            verbose=False,
            device=DEVICE
        )

        result = results[0]

        active = 0

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            active = len(ids)

            for box, tid in zip(
                boxes,
                ids
            ):

                tid = int(tid)

                if tid not in states:
                    states[tid] = PersonState(tid)

                state = states[tid]

                x1, y1, x2, y2 = box

                raw_x = (
                    x1 + x2
                ) / 2.0

                raw_y = y2

                raw_pos = (
                    raw_x,
                    raw_y
                )

                raw_speed = 0.0

                if (
                    state.last_raw_position
                    is not None
                    and state.last_seen_time
                    is not None
                ):

                    dt = (
                        current_time
                        - state.last_seen_time
                    )

                    if dt > 0:

                        raw_speed = float(
                            np.hypot(
                                raw_x
                                - state.last_raw_position[0],
                                raw_y
                                - state.last_raw_position[1]
                            ) / dt
                        )

                state.positions.append(
                    (
                        current_time,
                        raw_x,
                        raw_y
                    )
                )

                (
                    sx,
                    sy,
                    vx,
                    vy,
                    speed,
                    scalar_rls_speed
                ) = state.update_rls(
                    current_time,
                    raw_x,
                    raw_y,
                    raw_speed
                )

                direction = None

                if speed > 1e-6:

                    direction = float(
                        np.degrees(
                            np.arctan2(
                                vy,
                                vx
                            )
                        )
                    )

                state.speeds.append(
                    (
                        current_time,
                        speed
                    )
                )

                trend = local_speed_trend(
                    state.speeds,
                    current_time
                )

                # --------------------------------------------------------
                # REGION FROM EXISTING ROI CONFIG ONLY
                # --------------------------------------------------------

                raw_region, raw_roi = classify_region(
                    raw_pos,
                    rois,
                    entering_names,
                    inside_names
                )

                smooth_region, smooth_roi = classify_region(
                    (sx, sy),
                    rois,
                    entering_names,
                    inside_names
                )

                # ROI decision is made from the RLS trajectory position.
                # The raw detection is NOT allowed to override the trajectory
                # state; otherwise detector jitter can create false crossings.
                region = smooth_region
                region_roi = smooth_roi

                state.region_history.append(
                    (
                        current_time,
                        region,
                        region_roi
                    )
                )

                # Geometric proximity to the configured ENTERING ROI.
                # This lets us detect a slowdown while still OUT, which is
                # exactly what creates the INTERESTED - PASSED outcome.
                entry_distance, nearest_entry_roi = distance_to_any_roi(
                    (sx, sy),
                    rois,
                    entering_names
                )

                # --------------------------------------------------------
                # INITIAL TRAJECTORY STATE
                # --------------------------------------------------------

                decide_initial_state(
                    state
                )

                # --------------------------------------------------------
                # PERSONAL BASELINE
                #
                # Baseline is learned only while outside configured
                # ENTERING/IN ROIs. No FRONT concept is used.
                # --------------------------------------------------------

                if not state.baseline_locked:

                    if (
                        region == "OUT"
                        and speed >= BASELINE_MIN_SPEED
                    ):

                        state.baseline_samples.append(
                            (
                                current_time,
                                speed
                            )
                        )

                        state.baseline_samples = [
                            z
                            for z in state.baseline_samples
                            if current_time - z[0]
                            <= BASELINE_WINDOW_SECONDS
                        ]

                    if (
                        len(state.baseline_samples)
                        >= BASELINE_MIN_SAMPLES
                    ):

                        vals = [
                            v
                            for _, v
                            in state.baseline_samples
                        ]

                        state.baseline_speed = float(
                            np.median(vals)
                        )

                        state.baseline_locked = True

                # Fallback for a valid OUT track that has not yet locked
                # a baseline.
                if (
                    state.baseline_speed is None
                    and state.initial_state == "OUT"
                    and speed >= BASELINE_MIN_SPEED
                ):

                    state.baseline_samples.append(
                        (
                            current_time,
                            speed
                        )
                    )

                    recent = [
                        v
                        for t, v
                        in state.baseline_samples
                        if current_time - t
                        <= BASELINE_WINDOW_SECONDS
                    ]

                    if (
                        len(recent)
                        >= BASELINE_MIN_SAMPLES
                    ):

                        state.baseline_speed = float(
                            np.median(recent)
                        )

                        state.baseline_locked = True

                # --------------------------------------------------------
                # MOTION INTEREST
                #
                # This is independent of entry. Once interested, no
                # later entry check is disabled.
                # --------------------------------------------------------

                baseline = state.baseline_speed

                if (
                    baseline is not None
                    and baseline > 1e-6
                ):

                    speed_ratio = (
                        speed / baseline
                    )

                    reduction = max(
                        0.0,
                        1.0 - speed_ratio
                    )

                    state.max_speed_reduction = max(
                        state.max_speed_reduction,
                        reduction
                    )

                else:

                    speed_ratio = 1.0
                    reduction = 0.0

                slowdown_condition = (
                    state.initial_state == "OUT"
                    and not state.ignore_interest
                    and not state.interested
                    and region == "OUT"
                    and entry_distance <= INTEREST_PROXIMITY_PIXELS
                    and baseline is not None
                    and speed_ratio <= SLOWDOWN_RATIO
                    and trend <= SLOWDOWN_SLOPE
                )

                if slowdown_condition:

                    if state.slowdown_start is None:
                        state.slowdown_start = current_time

                    duration = (
                        current_time
                        - state.slowdown_start
                    )

                    state.max_slowdown_duration = max(
                        state.max_slowdown_duration,
                        duration
                    )

                    if (
                        duration
                        >= MIN_SLOWDOWN_DURATION
                    ):

                        state.interested = True

                        state.interest_reasons.add(
                            "sustained_slowdown"
                        )

                else:

                    state.slowdown_start = None

                stop_condition = (
                    state.initial_state == "OUT"
                    and not state.ignore_interest
                    and not state.interested
                    and region == "OUT"
                    and entry_distance <= INTEREST_PROXIMITY_PIXELS
                    and baseline is not None
                    and speed_ratio <= STOP_RATIO
                )

                if stop_condition:

                    if state.stop_start is None:
                        state.stop_start = current_time

                    duration = (
                        current_time
                        - state.stop_start
                    )

                    state.max_stop_duration = max(
                        state.max_stop_duration,
                        duration
                    )

                    if (
                        duration
                        >= MIN_STOP_DURATION
                    ):

                        state.interested = True

                        state.interest_reasons.add(
                            "near_stop"
                        )

                else:

                    state.stop_start = None

                # --------------------------------------------------------
                # ENTRY
                #
                # ALWAYS RUN for an initially-OUT track.
                # It does not care whether the person is already interested.
                # --------------------------------------------------------

                update_entry_state(
                    state
                )

                # --------------------------------------------------------
                # Update latest observation.
                # --------------------------------------------------------

                state.last_raw_position = raw_pos
                state.last_seen_time = current_time

                reasons = sorted(
                    state.interest_reasons
                )

                state.metrics.append({
                    "frame": frame_number,
                    "time": current_time,
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "smooth_x": sx,
                    "smooth_y": sy,
                    "vx": vx,
                    "vy": vy,
                    "raw_speed": raw_speed,
                    "speed": speed,
                    "speed_trend": trend,
                    "baseline_speed": (
                        baseline
                        if baseline is not None
                        else 0.0
                    ),
                    "speed_ratio": speed_ratio,
                    "region_status": region,
                    "region_roi": region_roi,
                    "entry_distance_px": entry_distance,
                    "nearest_entry_roi": nearest_entry_roi,
                    "initial_state": state.initial_state,
                    "initial_roi": state.initial_roi,
                    "entered": state.entered,
                    "entry_roi": state.entry_roi,
                    "interested": state.interested,
                    "ignored_initially_inside": state.ignore_interest,
                    "interest_reasons": reasons,
                })

                all_metrics[tid].append(
                    state.metrics[-1]
                )

                # --------------------------------------------------------
                # DRAW
                # --------------------------------------------------------

                cv2.rectangle(
                    output,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                draw_trajectory(
                    output,
                    [
                        (p[1], p[2])
                        for p in state.positions
                    ]
                )

                draw_arrow(
                    output,
                    (sx, sy),
                    direction
                )

                if state.ignore_interest:

                    status = (
                        "ALREADY IN - IGNORED"
                    )

                    status_color = (
                        150,
                        150,
                        150
                    )

                elif (
                    state.interested
                    and state.entered
                ):

                    status = (
                        "INTERESTED - ENTERED"
                    )

                    status_color = (
                        0,
                        255,
                        0
                    )

                elif state.interested:

                    status = (
                        "INTERESTED - PASSED"
                    )

                    status_color = (
                        0,
                        165,
                        255
                    )

                else:

                    status = (
                        "NOT INTERESTED"
                    )

                    status_color = (
                        200,
                        200,
                        200
                    )

                cv2.putText(
                    output,
                    f"ID {tid} | {status}",
                    (
                        int(x1),
                        max(
                            20,
                            int(y1) - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    status_color,
                    2
                )

                cv2.putText(
                    output,
                    f"Region: {region}",
                    (
                        int(x1),
                        int(y2) + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (
                        0,
                        255,
                        0
                    )
                    if region != "OUT"
                    else
                    (
                        180,
                        180,
                        180
                    ),
                    2
                )

                # User requested trend rather than speed on video.
                cv2.putText(
                    output,
                    f"Trend: {trend:.1f}",
                    (
                        int(x1),
                        int(y2) + 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2
                )

                if reasons:

                    cv2.putText(
                        output,
                        "Reason: "
                        + "+".join(reasons),
                        (
                            int(x1),
                            int(y2) + 60
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        status_color,
                        2
                    )

        # --------------------------------------------------------
        # GLOBAL OVERLAY
        # --------------------------------------------------------

        cv2.putText(
            output,
            f"Frame: {frame_number}/{frames_to_process}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Active tracks: {active}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        eligible_states = [
            s
            for s in states.values()
            if not s.ignore_interest
        ]

        live_interested = sum(
            s.interested
            for s in eligible_states
        )

        live_entered = sum(
            s.interested and s.entered
            for s in eligible_states
        )

        live_passed = sum(
            s.interested and not s.entered
            for s in eligible_states
        )

        cv2.putText(
            output,
            f"Interested: {live_interested} | "
            f"Entered: {live_entered} | "
            f"Passed: {live_passed}",
            (20, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        writer.write(
            output
        )

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{frames_to_process} "
                f"| Unique IDs: "
                f"{len(states)} "
                f"| Interested: "
                f"{live_interested} "
                f"| Entered: "
                f"{live_entered}"
            )

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    print(
        "\nWriting CSV files..."
    )

    write_frame_csv(
        all_metrics,
        FRAME_CSV
    )

    write_track_csv(
        states,
        DETAIL_CSV
    )

    write_summary(
        states,
        SUMMARY_CSV
    )

    print(
        "Generating diagnostic plots..."
    )

    plots = 0

    for tid, state in states.items():

        if (
            len(state.metrics)
            >= MIN_TRACK_SAMPLES_FOR_PLOT
        ):

            generate_track_plot(
                tid,
                state.metrics,
                state,
                PLOT_DIR
            )

            plots += 1

    generate_summary_pie_chart(
        states,
        "outputs/interest_outcomes_pie.png"
    )

    eligible = [
        s
        for s in states.values()
        if not s.ignore_interest
    ]

    interested = sum(
        s.interested
        for s in eligible
    )

    interested_entered = sum(
        s.interested and s.entered
        for s in eligible
    )

    interested_passed = sum(
        s.interested and not s.entered
        for s in eligible
    )

    ignored = sum(
        s.ignore_interest
        for s in states.values()
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "Behavioral analysis completed"
    )

    print(
        "=" * 72
    )

    print(
        f"Frames processed       : {frame_number}"
    )

    print(
        f"Unique people          : {len(states)}"
    )

    print(
        f"Ignored initially IN   : {ignored}"
    )

    print(
        f"Interested             : {interested}"
    )

    print(
        f"Interested + Entered   : "
        f"{interested_entered}"
    )

    print(
        f"Interested + Passed    : "
        f"{interested_passed}"
    )

    print(
        f"Plots                  : {plots}"
    )

    print(
        f"Output video           : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Per-track CSV          : "
        f"{DETAIL_CSV}"
    )

    print(
        f"Frame metrics CSV      : "
        f"{FRAME_CSV}"
    )

    print(
        f"Summary CSV            : "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Plots directory        : "
        f"{PLOT_DIR}"
    )

    print(
        "Interest pie chart      : "
        "outputs/interest_outcomes_pie.png"
    )


if __name__ == "__main__":
    main()
