"""
Pure ROI trajectory state machine.

Deliberately has NO dependency on cv2 / ultralytics so it can be unit
tested offline with synthetic trajectories before being wired into the
full 6000-frame video pipeline.

This module is imported by both:
  - behavioral_interest_analysis.py (the real pipeline)
  - test_state_machine.py (the offline synthetic test harness)

so the exact same code that runs on video runs in the tests.
"""

import numpy as np
from collections import deque


# ============================================================
# Tunable constants (all time-based, consistent with the rest
# of the pipeline's existing hysteresis constants such as
# MIN_SLOWDOWN_DURATION=0.50s / MIN_STOP_DURATION=0.75s).
# ============================================================

# A single noisy IN sample must never create an entry (Rule 9).
# We require the IN region to be observed continuously for at
# least this long before we commit to "entered". This is a small
# fraction of a second -- enough to reject one-frame detector/ROI
# jitter, while still recording the entry at the TRUE first-IN
# timestamp (not the confirmation timestamp).
ENTRY_CONFIRM_SECONDS = 0.15


# ============================================================
# RLS (unchanged from the original pipeline)
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


# ============================================================
# Per-person state
# ============================================================

class PersonState:

    def __init__(self, track_id, trail_length=300):
        self.track_id = int(track_id)

        self.positions = deque(maxlen=trail_length)
        self.speeds = deque(maxlen=trail_length)

        # (time, semantic region, ROI name) - ALL ROI observations,
        # never truncated. Classification MUST use this, not just the
        # current frame.
        self.region_history = []
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

        self.interested = False
        self.entered = False

        self.entry_time = None
        self.entry_roi = None

        self.interest_reasons = set()

        self.initial_state = None
        self.initial_roi = None
        self.initial_state_decided = False
        self.ignore_interest = False

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
# Initial trajectory state
#
# UNCHANGED from the original pipeline. No CSV evidence showed this
# to be broken (short 1-2 sample tracks that were 100% IN were
# correctly ignored). Per the "make the minimum change required"
# instruction, this is left as-is.
# ============================================================

def decide_initial_state(state):
    """
    Freeze eligibility from the FIRST trajectory ROI observation.

        first ROI = IN       -> already inside -> ignore forever
        first ROI = ENTERING -> interested + entered
        first ROI = OUT      -> valid candidate; entry can only happen later
    """
    if state.initial_state_decided:
        return

    if not state.region_history:
        return

    first_time, first_region, first_roi = state.region_history[0]
    state.initial_region_observed = first_region
    state.initial_state = first_region
    state.initial_roi = first_roi
    state.initial_state_decided = True

    if first_region in ("IN", "NA"):
        state.ignore_interest = True
        if first_region == "IN":
            state.interest_reasons.add("initially_inside")
        else:
            state.interest_reasons.add("initially_outside_all_rois")

    elif first_region == "ENTERING":
        state.interested = True
        state.entered = True
        state.entry_time = first_time
        state.entry_roi = first_roi
        state.entry_detected_from = "initial_trajectory"
        state.interest_reasons.add("initially_entering")

    elif first_region == "OUT":
        # Valid candidate.
        pass


# ============================================================
# Entry detection -- THE FIX
#
# BUG (confirmed from behavioral_interest_frame_metrics.csv, track
# IDs 362 and 370): the previous implementation scanned
# region_history[1:] and returned on the FIRST occurrence of EITHER
# "ENTERING" or "IN". That means merely touching the ENTERING ROI
# was enough to permanently lock entered=True, even though the
# person had not yet reached the IN ROI (and, per Rule 4/Rule 9,
# might turn back to OUT and never reach it at all).
#
# Concrete evidence:
#   track 362: ENTERING first touched at frame 5592 (t=186.37s).
#              entered flipped to 1 THAT SAME FRAME, with
#              entry_roi recorded as "inside_store_entering_area"
#              (the ENTERING roi) and entry_time=186.37s.
#              The person did not actually reach IN
#              ("inside_store") until frame 5612 (t=187.03s) --
#              almost 0.7s and 20 frames later, and even dipped back
#              to OUT for one frame (5611) in between.
#   track 370: identical pattern -- entered locked at the ENTERING
#              frame (184.27s), true IN not reached until 185.03s.
#
# This run happened to not produce a final misclassification only
# because every track that touched ENTERING in this dataset also
# eventually reached IN. But the mechanism is wrong: it commits to
# "entered" based on ENTERING, and it captures the wrong entry_roi
# and an artificially early entry_time. In a case where the person
# instead turned back (OUT -> ENTERING -> OUT), it would have
# produced exactly the false "Interested + Entered" bug described
# in the prompt.
#
# FIX: entry now requires the IN region specifically, sustained for
# ENTRY_CONFIRM_SECONDS (Rule 9 hysteresis, rejects one noisy frame),
# with the recorded entry_time/entry_roi taken from the FIRST sample
# of that confirmed run (the true crossing moment), not the
# confirmation moment. ENTERING is not, by itself, evidence of entry
# -- exactly as Rule 4 requires ("OUT -> ENTERING => candidate
# entry; continue observing trajectory").
# ============================================================

def find_confirmed_entry(state):
    """
    Scan the COMPLETE trajectory history (never truncated) for a
    persistent run of IN observations, for a track whose origin was
    OUT. Returns (timestamp, region, roi_name) of the START of that
    run, or None if no confirmed entry exists yet.

    Tolerates tracker/frame gaps and any amount of ENTERING <-> OUT
    wobble before the IN run -- only the accumulated duration of the
    IN run itself matters.
    """
    if state.initial_state != "OUT":
        return None

    history = state.region_history
    if len(history) < 2:
        return None

    run_start_time = None
    run_start_roi = None
    in_run_active = False

    for timestamp, region, roi_name in history[1:]:
        if region == "IN":
            if not in_run_active:
                run_start_time = timestamp
                run_start_roi = roi_name
                in_run_active = True

            duration = timestamp - run_start_time
            if duration >= ENTRY_CONFIRM_SECONDS:
                return run_start_time, "IN", run_start_roi
        else:
            # Any non-IN sample (OUT or ENTERING) breaks the run.
            # This is what makes OUT -> ENTERING -> OUT (Rule 4/5)
            # and a single noisy OUT-IN-OUT blip (Rule 9) correctly
            # NOT count as entry.
            in_run_active = False
            run_start_time = None
            run_start_roi = None

    return None


def update_entry_state(state):
    """Commit a permanent entry event from the confirmed trajectory history."""
    if state.ignore_interest or state.entered:
        return

    evidence = find_confirmed_entry(state)
    if evidence is None:
        return

    timestamp, region, roi_name = evidence
    state.entered = True
    state.entry_time = timestamp
    state.entry_roi = roi_name or region
    state.entry_detected_from = "trajectory_history"
    state.interest_reasons.add("entered_store")
    state.interested = True
