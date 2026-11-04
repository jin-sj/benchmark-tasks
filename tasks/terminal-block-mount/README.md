# Mount a terminal block next to an existing one (two arms, loose DIN rail)

An electrician's step in building a terminal strip. A loose 24 cm TS35 x 7.5 DIN rail lies on the table with an end
bracket near each end and a beige terminal block already mounted against the left bracket. One arm must hold the
rail down; the other takes a blue terminal block from an open parts tray, snaps it onto the rail by pressing straight
down (its spring feet click under the rail's lips), and slides it along the rail until it sits flush against the
beige block.

The blocks are shaped like a Weidmüller WDU 2.5 feed-through terminal. The latch is a spring mechanism standing in
for the real plastic flexures: a latched block is clamped to the rail, so it takes a push to slide it, it stays where
it is left, and it cannot be pulled straight off.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `tb_parts.py` | the custom parts (rail assembly, WDU-style blocks with the sprung latch, tray): geometry, USD writer, and the hook that lets yam_arena spawn them |
| `demo.py` | scripted two-arm demo, closed loop on the simulated object poses, and its variants |
| `latch_bench.py` | test bench for the latch alone: presses, slides, pulls and tilts latched blocks with measured forces |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the blocks stand touching in the tray and must start where the task puts them):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Hold the DIN rail down with one arm. With the other, take a blue terminal block from the tray, snap it onto the rail by pressing it straight down, and slide it along the rail until it sits flush against the beige block."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed (jitter off). Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one
GPU with the three cameras rendering (a full 1,200-step trial is about 7 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py --variant full                 # also: one_arm, no_press, no_slide
    python demo.py --variant all --no-video       # all four in one simulator session (~12 min on GPU)
    python demo.py --variant all --no-video --device cpu   # CPU physics, ~3x faster, for quick checks

It writes `result.json`, `steps.jsonl` (per-step stage, rail motion, block pose, feet, pads, gaps and speed), one
video per camera and `setup.png` to `--out`. Benchmark trials run GPU physics (`cuda:0`); CPU physics gives the same
scores for all four variants but differs in detail.

Latch bench (about 4 minutes): `python latch_bench.py [--device cpu]`, writes `latch_bench.json`.

## Scoring

All measured in the rail's own frame, best over the three blue blocks, at the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | success: a block **latched** and **flush**, and still, held for 1 s in a row, with the rail in place |
| 0.75 | a block **latched**: both feet clicked under the lips and the clamp engaged, i.e. the block's base within -0.5..+0.8 mm of the lip tops at both lip edges, both feet back in (spread under 1 mm), both seat pads squeezed by the lips (at least 0.5 mm), its centre within 1 mm of the rail centreline, square (yaw) and level (tilt) within 3 degrees, between the two end brackets |
| 0.5 | a block **resting on the lips**: between the brackets, centred within 4 mm, its base within 8 mm above the lip tops at both edges, tilt under 10 degrees (a block set on the rail unpressed sits about 4.5 mm high on its feet's lead-ins, or tips onto one lip edge with the other foot hooked) |
| 0.25 | a block **lifted**: its lowest point 25 mm or more above the table |
| 0 | otherwise |

- **Flush:** latched, and the gap to the beige block's face is 0.5 mm or less (and no more than 0.5 mm of overlap) at
  four points on the facing side: 15 mm either side of the centre, 0.5 mm and 38 mm above the lips. A pitched or
  turned block fails (0.5 mm over the 37.5 mm between the low and high points is 0.76 degrees).
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step (see Fidelity).
- **Rail in place:** the rail's origin moved less than 10 mm across the table and it turned less than 3 degrees from
  where it started. Otherwise success is impossible and the score is capped at 0.25.
- Measurable with a ruler (rail displacement, block height on the rail, centring), a feeler gauge (gaps) and a
  protractor or photo (angles).

No unintended strategy scores: a block set on the rail unpressed sits about 4.5 mm high and does not count as
latched; the latched height can only be reached with both teeth under the lips (the geometry collides otherwise),
and a block pressed down but not clicked has its feet spread or its pads unsqueezed; end brackets stand on the lips
at both ends of the working span, so a block cannot be slid on from a rail end; a block on the stub beyond the left
bracket is outside the span and earns no rail credit; moving the rail caps the score.

## Measured results

Physics at 240 Hz on GPU (`cuda:0`), 20 Hz control, object jitter 0, run limit 1,200 steps. The simulation is
deterministic here: a separate full run with video gave the identical result.

| variant | what the arms do | score | notes |
| --- | --- | --- | --- |
| `full` | left arm holds the rail; right arm picks, presses, slides | **1.0** (success) | clicked on at step 295; flush at step 522; success at step 540. Final gaps 0.11-0.26 mm, tilt 0.3 degrees, rail moved 0.0004 mm. The hand was gone within 4 steps of success and the block stayed flush, gaps unchanged, for the 3 s the demo kept running |
| `one_arm` | left arm stays in its ready pose; right arm runs the same script | **0.25** | the block clicked on, but sliding it dragged the unheld rail 106 mm and turned it 4 degrees (the demo stops 1 s after the rail leaves its limits) |
| `no_press` | block set on the lips and let go, never pressed | **0.5** | resting on its lead-ins, 4.5-5.1 mm above the lips |
| `no_slide` | pressed on and latched 36.5 mm short of the beige block, never slid | **0.75** | latched, not flush |

With CPU physics (`--device cpu`) the four variants score the same: 1.0 (success at step 574), 0.25, 0.5 and 0.75.

Latch bench (`latch_bench.py`, GPU; forces ramped on blocks on fixed rails):

| test | result |
| --- | --- |
| press straight down (also 0.5 mm and 1 mm off centre) | clicks at 5.2-5.3 N; an unpressed block rests 4.5 mm high |
| latched block, untouched | sits 0.07 mm from its nominal seat, level |
| push along the rail, 2.5 mm above the lips | starts sliding at 2.2 N (pitch before that 0.2 degrees); the loose rail on the table slides at 0.65 N |
| push it 3 mm, then let go | stops within 0.3 mm |
| pull straight up (to 40 N), tilt off (to 30 N) | does not come off |

## How the parts are built

yam_arena's task files only offer single primitives and library assets, and yam_arena has no extension point for
multi-shape or jointed parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified:

- `Part` subclasses yam_arena's `Box`, so the stock loader accepts it; `task.py` calls `tb_parts.install()`, which
  wraps `yam_arena.scene.task_object` (and `object_states`) the moment that module is imported (after Isaac Sim has
  started). Parts are then spawned from USD files that `tb_parts` writes to a temporary folder; every other object
  goes through yam_arena unchanged. Before wrapping, it checks both functions' signatures and stops with an error if
  yam_arena has changed them.
- **Rail** (one free rigid body, 150 g, steel friction 0.3): 1 mm web and side walls, lips 2 mm thick with outer
  edges at +-17.5 mm. Two tapered end brackets (8 mm thick, 44 mm wide at the base, 24 mm at the top, 37.5 mm above
  the lips) and the beige block are part of the same body.
- **Blue block** (an articulation of five bodies, about 8 g): the WDU 2.5 profile, 5.1 mm thick, 60 mm long across
  the rail, 46 mm tall (39.5 mm above the lips), taken from the manufacturer's catalogue drawing and CAD, simplified
  to about 0.5 mm and written into `tb_parts.py` as coordinates (no CAD file is read). It looks like the real part
  (stepped outline, screw heads in their shafts, wire entries, through-holes) and collides as 14 convex pieces.
- **The latch** (a spring hack, not a flexure model; flat-on-flat contacts, which PhysX resolves reliably):
  - *the click:* each foot is its own link on a prismatic joint across the rail, sprung inward at 1 N/mm, with a
    tooth reaching 2 mm under the lip, a small chamfer and a lead-in below. Pressing down, the lip edges spread the
    feet on their lead-ins until the teeth pass under the lips and snap back in.
  - *the clamp:* over each lip, a seat pad under the housing on a vertical joint sprung at 3 N/mm stands 1.2 mm proud
    when free. Pressing the block on squeezes the pads; let go, they push it back up until the teeth bear on the lips
    from below. A latched block therefore sits 0.3 mm above the lips without play, resists sliding (friction on the
    pads and under the lips), and does not rock.
  - the tooth tops are flat, as on the real terminal, so a block cannot be pulled straight off: a foot has to be
    spread first.
- **Beige block:** the same shape, part of the rail body; its housing collides and its feet are looks only.
- **Tray** (fixed): an open tray with 12 mm walls and no slots. The three blocks stand upright side by side, 5.2 mm
  apart, staggered 44 mm along their length so each leans on its neighbour over 16 mm while the middle of each end
  block is free on both faces for a grip across them.
- Arena's geometry queries don't cover articulations, so the block's `ObjectState` (bounding-box centre, orientation,
  velocities, bounds, plus the feet's spread and the pads' squeeze) is computed by `tb_parts`.

## The demo

Left arm: fingers closed, turned to close across the rail, leaned 34 degrees with the wrist away from the work,
pressed into the rail's channel in the free stub beyond the left bracket, 36 mm from the beige block, for the whole
run (commanded 8 mm into the web).

Right arm: grips the tray's block nearest the rail across its broad faces (pad tips 20 mm below its top), lifts it,
carries it over the rail 36 mm short of the beige block, aligns it on its measured pose, lowers it until its feet meet
the lips and presses straight down at 5 mm/s while still holding it (correcting across the rail until the feet reach
the lip edges) until the block bottoms on the lips; lets go; then pushes the block's side low down with closed
fingertips, the gripper leaned back 28 degrees and turned so it is narrow along the rail, at 8 mm/s and 3 mm/s for the
last 3 mm, shifting the push point toward whichever end lags to keep the block square, with a firmer push in the last
2 mm, until the gaps are under 0.3 mm; then at once backs off 8 mm, lifts clear and moves away.

The lean matters: the closed YAM fingers widen upward, and leaned less than about 25 degrees the upper finger touches
the block near its top before the tips do, which pitches it and jams it on the lips.

The two grippers are about 10 cm wide in their closing direction, which is why the hold point is in the stub 36 mm
beyond the beige block and the block is snapped on 36 mm away and slid that far: closer, the two grippers collided.

## Fidelity: what is simplified

- Lips 2 mm thick (a real TS35 rail's are 1 mm) so a pressed tooth cannot tunnel through them at 240 Hz.
- The latch is springs and rigid links, not flexing plastic, and its features are enlarged (2 mm tooth reach, 0.3 mm
  play) because PhysX's contact error at this scale is about 0.2-0.4 mm. It clicks on at about 5 N, likely well
  below a real block's snap-on force (not measured here). Sliding a latched block takes about 2 N.
- The feet and pads carry joint armature (0.05 kg) so the light parts on stiff springs don't ring at the 240 Hz
  physics rate. A side effect: while a block sits clamped, PhysX reports a steady velocity for it that it does not
  have (about 6 mm/s and 0.4 rad/s on GPU, more on CPU, with the pose unchanged to a micrometre). The block's
  `ObjectState` therefore reports its motion over the last control step, computed from its pose.
- Blocks keep their outline, screws, wire entries and through-holes as looks; collision uses 14 convex pieces. No
  working screw clamps, marker slots or cross-connection parts. The beige block and the end brackets are part of the
  rail body; the brackets have no clamping screws.
- The rail is 150 g lumped in one body.
- Friction: rail steel 0.3, block housing polyamide 0.35, feet and pads on the lips 0.25 (lower and an unpressed block
  slides off its lead-ins; higher and the feet jam), tray plastic 0.4, table and robot fingers PhysX default 0.5.
  Contact offsets 0.5 mm on the parts, rest offset 0.
- The tray is fixed to the table.

## Known weak points

- The scorer cannot tell whether a hand is still touching the block: a policy that holds the block flush and still
  for 1 s scores success. The demo lets go immediately and checks that the block stays flush on its own.
- A block set on the rail unpressed is not in a stable rest: it stays perched on its lead-ins or tips onto one lip
  edge depending on small disturbances (as a real one does without enough friction). The 0.5 credit covers both.
- The left gripper holds the rail leaned 34 degrees in the 43 mm stub beyond the left bracket, which the gripper
  widths force. It works but is an awkward hold.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error. It must run before `yam_arena.scene` is first imported (true for `-E task=task.py`).
- Object jitter must be 0: seeded jitter would shift the touching blocks independently into each other.
- The demo was validated through `demo.py`, which uses the same embodiment, task loader and per-step scoring as
  `inspect-robots run`. Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`),
  scored 0.0: in 1,200 steps its arms never reached the rail or the blocks.
