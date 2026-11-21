#!/usr/bin/env python3
"""
Lane Departure Warning -- ADAS lane keeping from a dashcam view
================================================================

A forward-facing dashcam watches the road ahead. Canny edges + Hough
line segments inside a road ROI are clustered into the left and right
lane boundaries, each fitted as a line (x as a function of y) and
smoothed over time, so brief dropouts do not lose the lane. From the
two fitted boundaries the system measures, every frame:

* the lane centre and the lane width at a reference height near the
  bottom of the frame (where the ego car "is");
* the vehicle's lateral offset from the lane centre in pixels and as
  a percentage of the lane width. The camera is almost never exactly
  over the car's centre on real dashcam footage, so the vehicle's
  lateral reference is CALIBRATED from the first 2 s of straight
  driving (the median lane centre over that window); offsets are
  deviations from that calibrated position, and the first 2 s read
  CALIBRATING;
* the state: CENTRED (|offset| < 12% of the lane width), DRIFTING
  LEFT / DRIFTING RIGHT, or DEPARTURE WARNING when the vehicle
  crosses a lane line -- the crossing must hold for ~0.5 s before
  the warning fires, and each crossing is logged once with a
  timestamp, per side.

The board shows frames processed, frames with both / one / no lane
found, mean and max lateral offset, drift entries and departure
events per side, and time spent in each state. The timestamped event
log is written to ``output_events.log`` and the final counters to
``output_summary.txt`` next to the video.

Simulation mode (``--simulate``) renders the same road from the
driver's seat -- asphalt with solid white edge lines on both sides --
and scripts the car drifting across the right line, back to centre,
then across the left line. The detector reads only the rendered
pixels; it never sees the scripted drift. Simulation is an internal
logic check; the delivered demo is real footage (see README).

Honest framing: lateral offset is measured in pixels relative to the
detected lane, not in surveyed metres -- the lane width in pixels
changes with perspective and with the road, so the percentage of lane
width is the portable number. A departure warning needs a real
crossing of a detected lane line; on footage where the car stays
centred, the warning is verified in simulation only. Faded, snow-
covered or missing markings, guardrails and shadows can all confuse
the line fit -- the README lists what the demo clip does and does
not show.

Examples
--------
    python lane_departure_warning.py                 (opens a file picker)
    python lane_departure_warning.py --source pick   (same picker, explicit)
    python lane_departure_warning.py --source road.mp4
    python lane_departure_warning.py --source 0      (webcam)
    python lane_departure_warning.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final counters
to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)
BLUE = (220, 160, 60)

# ---------------------------------------------------------------------------
# Lane detection tuning (video-area coordinates, 1280x524)
# ---------------------------------------------------------------------------
# Road ROI: a trapezoid from the bottom corners up to the vanishing
# point. Tuned for a forward dashcam: the road fills the lower half
# and converges near the centre of the frame.
ROI_POLY = np.array([[(20, VIDEO_H), (520, 200),
                      (760, 200), (1260, VIDEO_H)]], dtype=np.int32)
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESH = 50
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 40
# A segment belongs to a lane boundary if |dx/dy| (x as a function of
# y) is in this range. The lower bound is deliberately small (0.15):
# a lane line near the camera's forward axis is almost vertical in
# the image -- on the demo clip the centre dashes read about 0.36,
# and a 0.35 cut-off rejected them entirely, so the fit latched onto
# a tyre track instead (found during verification). Near-vertical
# posts and trees also pass this filter; the vanishing-point-anchored
# fit and the innermost-family selection are what separate them from
# the lane lines. Near-horizontal segments (horizon, cross markings)
# are still rejected.
SLOPE_MIN, SLOPE_MAX = 0.15, 2.0
# Temporal smoothing of the fitted lines (EMA) and how long a missing
# side is held before it is dropped (frames at 30 fps).
SMOOTH_ALPHA = 0.35
HOLD_FRAMES = 30
# Vanishing point of the road (video coordinates, 1280x524). The
# lane fits are anchored here: every lane line on a straight road
# passes through the vanishing point, so each fitted line is the
# length-weighted median of the lines joining the vanishing point to
# the segment endpoints. Anchoring at the vanishing point keeps the
# slope honest when the visible segments are short dashes clustered
# near the bottom of the frame -- a free least-squares fit over
# those alone reads far too steep (measured during verification:
# -1.48 instead of about -0.77, drawing an X across the road).
# The vanishing point is a per-scene setting: measure it on the clip
# (where the lane lines converge) and pass --vp-x/--vp-y. The default
# (640, 190) suits a centred dashcam; the demo clip's measured VP
# is set in the README.
VP_X, VP_Y = 640.0, 190.0
# Reference height where the offset is measured (near the bottom of
# the frame, where the ego car "is").
Y_REF = 500
# State thresholds, as fractions of the lane width at Y_REF.
CENTRED_FRAC = 0.12               # |offset| below this: CENTRED
# Departure: the vehicle centre reaches a lane line, i.e. |offset|
# reaches half the lane width. Must hold this long (seconds) first.
DEPARTURE_HOLD_S = 0.5
# State changes are debounced: a new state must hold this many frames
# (0.3 s) before the board switches, so a noisy offset hovering on a
# threshold does not flicker CENTRED <-> DRIFTING every few frames.
STATE_HOLD_FRAMES = 9

VEHICLE_X = VIDEO_W // 2          # fallback if calibration finds nothing
# Calibration: the camera is almost never exactly over the car's
# centre on real dashcam footage (it sits left of centre, or the
# lens is wide), so the vehicle's lateral reference is CALIBRATED
# from the first seconds of straight driving: the median lane
# centre over that window is where the car "sits". Offsets are
# then deviations from that calibrated position -- the same
# calibration a production LDW does at start-up. Until the
# calibration window closes, no drift/departure is judged.
CALIB_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Lane tracker
# ---------------------------------------------------------------------------
class LaneTracker:
    """Fits and smooths the two lane boundaries; derives the state."""

    def __init__(self):
        self.left = None          # (a, b): x = a*y + b
        self.right = None
        self.left_age = 0
        self.right_age = 0
        self.frames = 0
        self.both_lanes = 0
        self.one_lane = 0
        self.no_lane = 0
        self.offsets = []         # signed offset in px, per frame
        self.offset_fracs = []    # offset as a fraction of lane width
        self.width_hist = []      # recent measured lane widths (px)
        self.nominal_width = 560.0  # fallback until a width is measured
        self.state = "NO LANE"
        self.state_time = {"CENTRED": 0.0, "DRIFTING LEFT": 0.0,
                           "DRIFTING RIGHT": 0.0, "DEPARTURE WARNING": 0.0,
                           "NO LANE": 0.0, "CALIBRATING": 0.0}
        # Camera calibration (see CALIB_SECONDS): the vehicle's
        # lateral reference, set from the first seconds of driving.
        self.vehicle_x = VEHICLE_X
        self.calib_centres = []
        self.calibrated = False
        self.drift_left_entries = 0
        self.drift_right_entries = 0
        self.depart_left_events = 0
        self.depart_right_events = 0
        self.depart_streak = 0    # frames the departure condition held
        self.depart_side = None   # side currently in a departure
        self.depart_logged = False
        self._pending = 0         # frames the new state has held
        self.events = []          # (t, text)
        self.video_frames = 0

    # -- fitting ------------------------------------------------------
    def _fit_side(self, segments, side):
        """Vanishing-point-anchored fit of a lane boundary, x = a*y + b.

        Only the INNERMOST line family is fitted: on the left the
        boundary closest to the vehicle (largest x at Y_REF), on the
        right the closest one (smallest x at Y_REF). A guardrail runs
        parallel to the edge line a little further out and would
        otherwise drag the fit away from the road.
        """
        if not segments:
            return None
        x_refs = []
        for (x1, y1, x2, y2, w) in segments:
            if y2 == y1:
                continue
            slope = (x2 - x1) / (y2 - y1)
            x_refs.append(x1 + (Y_REF - y1) * slope)
        if not x_refs:
            return None
        # Innermost family: on the left the boundary closest to the
        # vehicle (largest x at Y_REF), on the right the closest one
        # (smallest x at Y_REF). A guardrail runs parallel to the edge
        # line a little further out and would otherwise drag the fit
        # away from the road; the 75th/25th percentile picks the
        # family nearest the vehicle.
        if side == "L":
            target = float(np.percentile(x_refs, 75))
        else:
            target = float(np.percentile(x_refs, 25))
        keep = [(s, xr) for s, xr in zip(segments, x_refs)
                if abs(xr - target) <= 80]
        if not keep:
            keep = list(zip(segments, x_refs))
        # Vanishing-point-anchored fit: each segment contributes the
        # slope of the line joining the vanishing point to the
        # segment's position at Y_REF (its x_ref, already computed
        # above) -- a length-weighted median of those slopes. Using
        # x_ref instead of raw endpoints keeps the fit honest when a
        # segment sits near the vanishing point, where a raw
        # endpoint's slope is nearly 0 and would drag the median
        # flat (measured during verification: -0.37 instead of
        # about -0.77, putting the line in the middle of the lane).
        # The median over segments is robust to the two parallel
        # line families (dashed centre line + solid edge line) that
        # the innermost selection cannot fully separate.
        slopes, weights = [], []
        for (x1, y1, x2, y2, w), xr in keep:
            slopes.append((xr - VP_X) / (Y_REF - VP_Y))
            weights.append(w)
        if not slopes:
            return None
        order = np.argsort(slopes)
        s_sorted = np.array(slopes)[order]
        w_sorted = np.array(weights, dtype=float)[order]
        cum = np.cumsum(w_sorted)
        a = float(s_sorted[np.searchsorted(cum, cum[-1] / 2)])
        if abs(a) < 0.05 or abs(a) > 2.5:
            return None
        return (a, VP_X - a * VP_Y)

    def _smooth(self, old, new):
        if old is None:
            return new
        return (SMOOTH_ALPHA * new[0] + (1 - SMOOTH_ALPHA) * old[0],
                SMOOTH_ALPHA * new[1] + (1 - SMOOTH_ALPHA) * old[1])

    def _segments(self, frame):
        """Canny + Hough inside the road ROI, split into left/right."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, ROI_POLY, 255)
        roi = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, HOUGH_THRESH,
                                minLineLength=HOUGH_MIN_LEN,
                                maxLineGap=HOUGH_MAX_GAP)
        left, right = [], []
        if lines is not None:
            for (x1, y1, x2, y2) in lines.reshape(-1, 4):
                if y1 == y2:
                    continue
                slope = (x2 - x1) / (y2 - y1)   # dx/dy
                if not (SLOPE_MIN <= abs(slope) <= SLOPE_MAX):
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                if length < HOUGH_MIN_LEN:
                    continue
                mx = (x1 + x2) / 2
                if slope < 0:
                    left.append((x1, y1, x2, y2, length))
                else:
                    right.append((x1, y1, x2, y2, length))
        return left, right

    # -- per-frame update ----------------------------------------------
    def update(self, frame, t):
        self.frames += 1
        left_segs, right_segs = self._segments(frame)
        new_left = self._fit_side(left_segs, "L")
        new_right = self._fit_side(right_segs, "R")

        if new_left is not None:
            # Temporal gate: a fit that jumps more than 60 px at Y_REF
            # in one frame is a passing car, glare or a shadow, not
            # the lane moving -- hold the previous fit instead. A real
            # lane change or curve moves the lines a few px per frame.
            if self.left is not None and abs(
                    (new_left[0] * Y_REF + new_left[1])
                    - (self.left[0] * Y_REF + self.left[1])) > 60:
                new_left = None
        if new_right is not None:
            if self.right is not None and abs(
                    (new_right[0] * Y_REF + new_right[1])
                    - (self.right[0] * Y_REF + self.right[1])) > 60:
                new_right = None

        if new_left is not None:
            self.left = self._smooth(self.left, new_left)
            self.left_age = 0
        else:
            self.left_age += 1
            if self.left_age > HOLD_FRAMES:
                self.left = None
        if new_right is not None:
            self.right = self._smooth(self.right, new_right)
            self.right_age = 0
        else:
            self.right_age += 1
            if self.right_age > HOLD_FRAMES:
                self.right = None

        both = self.left is not None and self.right is not None
        if both:
            self.both_lanes += 1
        elif self.left is not None or self.right is not None:
            self.one_lane += 1
        else:
            self.no_lane += 1

        offset = None
        frac = None
        half_w = None
        width = None
        centre = None
        if both:
            xl = self.left[0] * Y_REF + self.left[1]
            xr = self.right[0] * Y_REF + self.right[1]
            width = xr - xl
            if width > 120:
                self.width_hist.append(width)
                if len(self.width_hist) > 90:
                    self.width_hist.pop(0)
                self.nominal_width = float(np.median(self.width_hist))
                centre = (xl + xr) / 2
        elif self.left is not None or self.right is not None:
            # One boundary only: the car can still be judged against
            # the visible line, using the recent median lane width
            # (a real LDW warns off a single line the same way).
            width = self.nominal_width
            if self.left is not None:
                xl = self.left[0] * Y_REF + self.left[1]
                centre = xl + width / 2
            else:
                xr = self.right[0] * Y_REF + self.right[1]
                centre = xr - width / 2

        # -- calibration: the first CALIB_SECONDS of straight driving
        # set the vehicle's lateral reference (median lane centre).
        if not self.calibrated:
            if both and centre is not None and t < CALIB_SECONDS:
                self.calib_centres.append(centre)
            if t >= CALIB_SECONDS:
                if self.calib_centres:
                    self.vehicle_x = float(np.median(self.calib_centres))
                self.calibrated = True
                self.events.append(
                    (t, f"calibrated: vehicle x = {self.vehicle_x:.0f} px"))

        if centre is not None and width:
            half_w = width / 2
            if self.calibrated:
                offset = self.vehicle_x - centre
                frac = offset / width
                self.offsets.append(offset)
                self.offset_fracs.append(frac)

        # -- state machine (debounced: a new state must hold 0.3 s) --
        if not self.calibrated:
            raw = "CALIBRATING"
            self.depart_streak = 0
            self.depart_side = None
            self.depart_logged = False
        elif offset is None:
            raw = "NO LANE"
            self.depart_streak = 0
            self.depart_side = None
            self.depart_logged = False
        else:
            if abs(frac) < CENTRED_FRAC:
                raw = "CENTRED"
                self.depart_streak = 0
                self.depart_side = None
                self.depart_logged = False
            elif offset > 0:
                raw = "DRIFTING RIGHT"
                if offset >= half_w:
                    if self._departure("RIGHT", t):
                        raw = "DEPARTURE WARNING"
                else:
                    self.depart_streak = 0
                    self.depart_side = None
                    self.depart_logged = False
            else:
                raw = "DRIFTING LEFT"
                if offset <= -half_w:
                    if self._departure("LEFT", t):
                        raw = "DEPARTURE WARNING"
                else:
                    self.depart_streak = 0
                    self.depart_side = None
                    self.depart_logged = False

        if raw != self.state:
            self._pending += 1
            if self._pending >= STATE_HOLD_FRAMES:
                prev = self.state
                self.state = raw
                self._pending = 0
                if self.state == "DRIFTING LEFT":
                    self.drift_left_entries += 1
                    self.events.append((t, "DRIFTING LEFT"))
                elif self.state == "DRIFTING RIGHT":
                    self.drift_right_entries += 1
                    self.events.append((t, "DRIFTING RIGHT"))
                elif self.state == "CENTRED" and prev in (
                        "DRIFTING LEFT", "DRIFTING RIGHT",
                        "DEPARTURE WARNING"):
                    self.events.append((t, "back CENTRED"))
        else:
            self._pending = 0

        self.state_time[self.state] = self.state_time.get(self.state, 0) + 1 / 30
        return {"offset": offset, "frac": frac, "half_w": half_w,
                "state": self.state, "t": t,
                "left_segs": len(left_segs), "right_segs": len(right_segs)}

    def _departure(self, side, t):
        """Departure condition held -- count the sustain, then fire.

        Returns True once the warning is active (the caller then
        reports the raw state as DEPARTURE WARNING)."""
        if self.depart_side != side:
            self.depart_side = side
            self.depart_streak = 0
            self.depart_logged = False
        self.depart_streak += 1
        if self.depart_streak >= int(DEPARTURE_HOLD_S * 30):
            if not self.depart_logged:
                self.depart_logged = True
                if side == "LEFT":
                    self.depart_left_events += 1
                else:
                    self.depart_right_events += 1
                self.events.append(
                    (t, f"DEPARTURE WARNING -- crossed {side} lane line"))
            return True
        return False

    # -- reporting ------------------------------------------------------
    def summary(self):
        n = max(1, self.frames)
        offs = self.offsets
        mean_off = float(np.mean(np.abs(offs))) if offs else 0.0
        max_off = float(np.max(np.abs(offs))) if offs else 0.0
        fracs = self.offset_fracs
        mean_frac = float(np.mean(np.abs(fracs))) if fracs else 0.0
        max_frac = float(np.max(np.abs(fracs))) if fracs else 0.0
        lines = [
            "Lane Departure Warning -- summary",
            f"frames: {self.frames} ({self.video_frames} written, "
            f"{self.frames / 30:.1f}s at 30 fps)",
            f"calibrated vehicle x: {self.vehicle_x:.0f} px "
            f"({len(self.calib_centres)} centres in the first "
            f"{CALIB_SECONDS:.0f} s)",
            f"both lanes found: {self.both_lanes} "
            f"({100 * self.both_lanes / n:.1f}%)",
            f"one lane only: {self.one_lane} "
            f"({100 * self.one_lane / n:.1f}%)",
            f"no lane: {self.no_lane} ({100 * self.no_lane / n:.1f}%)",
            f"mean |lateral offset|: {mean_off:.1f} px "
            f"({100 * mean_frac:.1f}% of lane width)",
            f"max |lateral offset|: {max_off:.1f} px "
            f"({100 * max_frac:.1f}% of lane width)",
            f"drift entries: LEFT {self.drift_left_entries}, "
            f"RIGHT {self.drift_right_entries}",
            f"departure events: LEFT {self.depart_left_events}, "
            f"RIGHT {self.depart_right_events}",
            "time in state: " + ", ".join(
                f"{k} {v:.1f}s" for k, v in self.state_time.items()),
            f"final state: {self.state}",
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation -- rendered road, scripted drift, same pixel pipeline
# ---------------------------------------------------------------------------
class RoadSim:
    """Renders a driver's-seat road; the car drifts across both lines.

    The lateral position p is in units of half lane width: 0 = centred,
    +1 = the vehicle centre sits on the right lane line, -1 on the
    left line. The renderer only draws pixels; the detector never sees p.
    """

    def __init__(self, w, h, fps):
        self.w, self.h = w, h
        self.fps = fps
        self.t = 0.0

    def _p(self, t):
        """Scripted lateral position over the 30 s run."""
        def ramp(t0, t1, a, b):
            if t < t0:
                return a
            if t > t1:
                return b
            u = (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)          # smoothstep
            return a + (b - a) * u
        if t < 5:
            return 0.0
        if t < 14:
            return ramp(5, 12, 0.0, 1.15) if t < 12 else 1.15
        if t < 18:
            return ramp(14, 18, 1.15, 0.0)
        if t < 20:
            return 0.0
        if t < 28:
            return ramp(20, 26, 0.0, -1.15) if t < 26 else -1.15
        return ramp(28, 30, -1.15, 0.0)

    def frame(self):
        t = self.t
        self.t += 1 / self.fps
        p = self._p(t)
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Sky and verges.
        img[:190] = (200, 205, 210)          # pale sky
        img[190:] = (70, 95, 60)             # grass verges
        # Road polygon: vanishing point shifts slightly with the drift,
        # the bottom edge shifts by the full lateral offset.
        vp_x = int(640 - p * 60)
        shift = int(p * 290)                 # px per half lane width
        road = np.array([[(150 - shift, self.h), (vp_x - 30, 190),
                          (vp_x + 30, 190), (1130 - shift, self.h)]],
                        dtype=np.int32)
        cv2.fillPoly(img, road, (52, 52, 52))
        # Lane lines: left edge, right edge (solid white). No centre
        # line inside the ego lane -- a dashed line in the middle of
        # the lane would be picked up as a second left boundary.
        # Bottom x positions at p=0: left 350, right 930.
        def line(x_bottom, x_top, colour, thick):
            cv2.line(img, (int(x_bottom), self.h), (int(x_top), 190),
                     colour, thick, cv2.LINE_AA)
        line(350 - shift, vp_x - 42, (235, 235, 235), 6)
        line(930 - shift, vp_x + 42, (235, 235, 235), 6)
        # A few roadside posts for texture (never on the road).
        for i, x in enumerate((90, 1210, 60, 1240)):
            y = 300 + (i % 2) * 120
            cv2.line(img, (x, y), (x, y + 46), (180, 180, 180), 4)
        # Light asphalt noise so the frame is not perfectly flat.
        noise = np.random.default_rng(int(t * 30)).integers(
            -6, 7, size=(self.h, self.w, 1)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_canvas(video_frame, tracker, res, t, mode_label):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    canvas[TITLE_H:TITLE_H + VIDEO_H] = video_frame
    # Title bar.
    cv2.putText(canvas, "LANE DEPARTURE WARNING  --  ADAS",
                (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, mode_label, (W - 330, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)

    # Lane overlay on the video area.
    ov = canvas[TITLE_H:TITLE_H + VIDEO_H]
    state = res["state"]
    line_colour = (GREEN if state == "CENTRED" else
                   AMBER if state.startswith("DRIFTING") else
                   RED if state == "DEPARTURE WARNING" else GREY)
    for side, fit in (("L", tracker.left), ("R", tracker.right)):
        if fit is None:
            continue
        a, b = fit
        # Lines are drawn from the vanishing point down: above the
        # VP a fitted line extrapolates to the WRONG side of the
        # frame, drawing an X across the road (found during
        # verification -- the first renders drew from y=200 while
        # the demo clip's VP sits at y=291).
        y0, y1 = int(VP_Y), VIDEO_H
        cv2.line(ov, (int(a * y0 + b), y0), (int(a * y1 + b), y1),
                 line_colour, 4, cv2.LINE_AA)
    if tracker.left is not None and tracker.right is not None:
        xl = tracker.left[0] * Y_REF + tracker.left[1]
        xr = tracker.right[0] * Y_REF + tracker.right[1]
        cx = int((xl + xr) / 2)
        for y in range(200, VIDEO_H, 24):
            cv2.line(ov, (cx, y), (cx, y + 12), YELLOW, 2, cv2.LINE_AA)
        cv2.line(ov, (int(xl), Y_REF), (int(xr), Y_REF), YELLOW, 1,
                 cv2.LINE_AA)
    # Vehicle centre marker at the bottom (calibrated position).
    vx = int(tracker.vehicle_x)
    cv2.drawMarker(ov, (vx, VIDEO_H - 12),
                   (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 22, 2,
                   cv2.LINE_AA)
    cv2.putText(ov, "EGO", (vx - 18, VIDEO_H - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # State banner.
    banner = state
    bcol = (GREEN if state == "CENTRED" else
            AMBER if state.startswith("DRIFTING") else
            RED if state == "DEPARTURE WARNING" else GREY)
    cv2.rectangle(ov, (16, 12), (420, 52), (20, 20, 24), -1)
    cv2.putText(ov, banner, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                bcol, 2, cv2.LINE_AA)
    if res["offset"] is not None:
        cv2.putText(ov, f"offset {res['offset']:+.0f} px "
                        f"({100 * res['frac']:+.0f}% of lane)",
                    (440, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
                    cv2.LINE_AA)

    # Bottom panel: counters + recent events.
    py = TITLE_H + VIDEO_H + 8
    cv2.putText(canvas, f"t = {t:5.1f}s    frames {tracker.frames}    "
                        f"both lanes {tracker.both_lanes} "
                        f"({100 * tracker.both_lanes / max(1, tracker.frames):.0f}%)",
                (16, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1,
                cv2.LINE_AA)
    offs = tracker.offsets
    mean_off = float(np.mean(np.abs(offs))) if offs else 0.0
    max_off = float(np.max(np.abs(offs))) if offs else 0.0
    cv2.putText(canvas, f"mean |offset| {mean_off:.0f} px    "
                        f"max |offset| {max_off:.0f} px    "
                        f"drift L {tracker.drift_left_entries}  "
                        f"R {tracker.drift_right_entries}    "
                        f"departures L {tracker.depart_left_events}  "
                        f"R {tracker.depart_right_events}",
                (16, py + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1,
                cv2.LINE_AA)
    # Offset bar: lane width scaled to 360 px, centred on the panel.
    bar_cx, bar_y = 640, py + 92
    half = 180
    cv2.rectangle(canvas, (bar_cx - half, bar_y - 10),
                  (bar_cx + half, bar_y + 10), (70, 74, 88), -1)
    cv2.line(canvas, (bar_cx - half, bar_y - 14),
             (bar_cx - half, bar_y + 14), RED, 3)
    cv2.line(canvas, (bar_cx + half, bar_y - 14),
             (bar_cx + half, bar_y + 14), RED, 3)
    cv2.line(canvas, (bar_cx, bar_y - 14), (bar_cx, bar_y + 14),
             GREY, 2)
    if res["offset"] is not None and res["half_w"]:
        px = max(-half, min(half, int(res["offset"] / res["half_w"] * half)))
        cv2.circle(canvas, (bar_cx + px, bar_y), 9, line_colour, -1)
    cv2.putText(canvas, "LEFT", (bar_cx - half - 52, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT", (bar_cx + half + 12, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "lane position (dot = ego car, red ticks = lane lines)",
                (16, py + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1,
                cv2.LINE_AA)
    # Recent events, right side of the panel.
    for i, (et, text) in enumerate(tracker.events[-3:]):
        cv2.putText(canvas, f"{et:5.1f}s  {text}",
                    (860, py + 18 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, AMBER if "DEPARTURE" in text else GREY, 1,
                    cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------------
def pick_source():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                   ("All files", "*.*")])
    root.destroy()
    if not path:
        print("No file chosen -- exiting.")
        sys.exit(0)
    return path


def open_source(source):
    if source == "pick":
        source = pick_source()
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VP_X, VP_Y
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered road instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--vp-x", type=float, default=VP_X,
                    help="vanishing-point x in video coordinates (1280x524)")
    ap.add_argument("--vp-y", type=float, default=VP_Y,
                    help="vanishing-point y in video coordinates (1280x524)")
    args = ap.parse_args()

    VP_X, VP_Y = args.vp_x, args.vp_y

    fps = 30
    tracker = LaneTracker()
    sim = RoadSim(VIDEO_W, VIDEO_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))   # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            raw = sim.frame()
            det_frame = raw
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        res = tracker.update(det_frame, t)
        canvas = draw_canvas(det_frame, tracker, res, t, mode_label)
        writer.write(canvas)
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Lane Departure Warning", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for et, text in tracker.events:
            f.write(f"{et:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
