# Lane Departure Warning

An ADAS lane-departure warning from a forward dashcam view: Canny
edges + Hough line segments inside a road ROI are clustered into the
left and right lane boundaries, each fitted as a line anchored at the
road's vanishing point and smoothed over time, so brief dropouts do
not lose the lane. From the two fitted boundaries the system measures
the lane centre and the lane width at a reference height near the
bottom of the frame, and the vehicle's lateral offset from the lane
centre in pixels and as a percentage of the lane width.

States: **CENTRED** (|offset| < 12% of the lane width),
**DRIFTING LEFT / DRIFTING RIGHT**, or **DEPARTURE WARNING** when
the vehicle crosses a lane line — the crossing must hold for ~0.5 s
before the warning fires, and each crossing is logged once with a
timestamp, per side. A state change must hold 0.3 s before the board
switches, so a noisy offset hovering on a threshold does not flicker.
If only one boundary is visible, the car is still judged against
that line using the recent median lane width. The board shows frames
processed, frames with both / one / no lane found, mean and max
lateral offset, drift entries and departure events per side, and time
spent in each state. The timestamped event log is written to
`output_events.log` and the final counters to `output_summary.txt`.

## Camera calibration

The camera is almost never exactly over the car's centre on real
dashcam footage (it sits left of centre, or the lens is wide), so
the vehicle's lateral reference is **calibrated** from the first 2 s
of straight driving: the median lane centre over that window is
where the car "sits", and offsets are deviations from that
calibrated position — the same calibration a production LDW does at
start-up. The first 2 s read CALIBRATING and no drift or departure
is judged in that window. If the calibration window finds no lane,
the vehicle reference falls back to the frame centre (640 px).

The vanishing point is a per-scene setting: measure where the lane
lines converge on the clip and pass `--vp-x` / `--vp-y` (video
coordinates, 1280x524). The default (640, 190) suits a centred
dashcam; the demo clip's measured vanishing point is (640, 291).

## How to run

```bash
pip install -r requirements.txt
python lane_departure_warning.py                 # opens a file picker
python lane_departure_warning.py --source road.mp4
python lane_departure_warning.py --source 0      # webcam
python lane_departure_warning.py --simulate      # rendered road
python lane_departure_warning.py --source road.mp4 --vp-x 640 --vp-y 291
```

Press Q to quit. The video is written to `output.mp4`, the
timestamped event log to `output_events.log` and the final counters
to `output_summary.txt` next to it.

## Demo — real footage (the delivered demo)

Free-license Pexels footage "cargo countryside dashcam driver
perspective" (video 37930336, forest road in snow, 36.4 s clip,
first 30 s, no looping, downloaded fresh for this build under the
14 Nov fresh-video rule — never used by any earlier project).
Vanishing point measured on the clip: (640, 291).

- 900 frames (30.0 s at 30 fps), both lanes found 900 (100.0%),
  one lane only 0, no lane 0
- Calibrated vehicle x: 688 px (60 lane centres in the first 2 s)
- Mean |lateral offset|: 23.9 px (5.8% of lane width); max 94.1 px
  (25.8% of lane width)
- Drift entries: LEFT 0, RIGHT 1 (at 25.4 s, held to the end of the
  clip as the road bends left and the car sits right in its lane)
- Departure events: LEFT 0, RIGHT 0 — the car never crosses a lane
  line on this clip
- Time in state: CENTRED 23.1 s, DRIFTING RIGHT 4.6 s, CALIBRATING
  2.0 s, NO LANE 0.3 s (the first frames, before the first fit)
- Final state: DRIFTING RIGHT
- Frames inspected at 5 s, 15 s and 28 s — the overlay lines sit on
  the painted centre dashes (left) and on the road edge (right);
  the lane-centre dashes and the EGO marker at the calibrated
  position are visible throughout.

Demo: `files/project-demos/lane-departure-warning-demo.mp4`
(+ H.264 LinkedIn copy in `linkedin-h264/`, codec-verified h264,
1280x720, 30 fps, 30.0 s).

## Simulation (internal logic check only — NOT the demo)

Rendered road from the driver's seat; the car drifts across the
right line, back to centre, then across the left line. The detector
reads only the rendered pixels; it never sees the scripted drift.

- 900 frames (30.0 s), calibrated vehicle x: 641 px
- Both lanes found 552 (61.3%), one lane only 348 (38.7%) — during
  a deep drift one line leaves the road ROI, and the one-lane
  fallback keeps the offset going
- Mean |lateral offset|: 148.2 px (27.0% of lane width); max
  309.1 px (56.7%)
- Drift entries: LEFT 2, RIGHT 2; departure events: RIGHT 1 (fired
  at 11.0 s), LEFT 1 (fired at 25.3 s)
- Time in state: CENTRED 10.3 s, DRIFTING LEFT 4.5 s, DRIFTING
  RIGHT 5.8 s, DEPARTURE WARNING 7.2 s, CALIBRATING 2.0 s

## Honest limits

- Lateral offset is in **pixels relative to the detected lane**,
  not surveyed metres — the lane width in pixels changes with
  perspective and with the road, so the percentage of lane width is
  the portable number.
- The **departure warning itself is verified in simulation only**:
  the real clip contains no lane crossing, so no DEPARTURE WARNING
  ever fires on the real footage. What the real footage proves is
  the lane tracking (both lanes 100%), the offset measurement and
  the drift state.
- On the demo clip the **right boundary is read off the
  asphalt/snow edge**, not the painted edge dashes: the painted
  dashes are too small and faint for the edge detector in flat
  overcast light, while the snow edge is strong and runs parallel
  to them about 70 px outboard. The lane width is therefore
  overestimated by that much and the left boundary (the painted
  centre dashes) is the anchor the offset really rests on.
- Faded, snow-covered or missing markings, guardrails and shadows
  can all confuse the line fit — the innermost-family selection
  and the temporal gate below exist for exactly that, and a line
  that jumps more than 60 px at the reference height in one frame
  is rejected as a passing car, glare or shadow.
- The offset is a deviation from the **calibrated** straight-driving
  position, not an absolute measurement against the surveyed lane
  centre: if the car is already hugging a line when the 2 s
  calibration runs, that position becomes "centred".

## Bugs found and fixed during verification

1. A free least-squares lane fit over short dash segments clustered
   near the bottom of the frame read far too steep (−1.48 instead of
   about −0.77) and drew an X across the road — the fit is now
   anchored at the vanishing point (length-weighted median of the
   slopes from the VP to each segment's position at the reference
   height).
2. That anchored fit first used raw segment endpoints, which dragged
   the slope flat (−0.37) for segments near the vanishing point —
   it now uses each segment's extrapolated position at the reference
   height instead.
3. The state machine flickered CENTRED ↔ DRIFTING (18 left-drift
   entries in the first real run) with the offset hovering on the
   threshold — state changes now need to hold 0.3 s (9 frames).
4. A guardrail running parallel to the edge line dragged the fit
   away from the road — the fit now uses only the innermost line
   family per side (75th percentile of x-refs on the left, 25th on
   the right), and a fit that jumps more than 60 px at the reference
   height in one frame is rejected (passing cars, glare, shadows).
5. Fitted lines were drawn from y=200, above the demo clip's
   vanishing point (y=291), so they extrapolated to the wrong side
   of the frame and crossed — lines are now drawn from the vanishing
   point down.
6. The first clip tried for this build (Pexels 37930365, dashcam on
   a divided highway with a central barrier) was **rejected** after
   frame inspection: its ego lane's left boundary is a barrier line
   the detector cannot see, so the left fit latched onto the
   oncoming lane's line instead, and the camera sits at the lane's
   left edge on a curving road — the wrong geometry for a
   centred-vehicle demo. Eight other candidates were downloaded and
   frame-inspected first and rejected (California highway 29539184:
   left line drops out entirely at several timestamps; sunset rural
   road 37909698: centre dashes too sparse, 0–5 segments; foggy
   divided highway 37930376: low contrast, curved; highway with the
   ego bonnet in frame 5382495; pickup truck + shadow 4372785;
   city street with Botts' dots 3602494; vertical clip 13229385).
   Their files were deleted; only the chosen forest-road clip was
   kept.
