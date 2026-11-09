# Find the terminal block that is not latched on a DIN rail and press it home

Five identical terminal blocks sit side by side on a DIN rail. Four are latched. One was hooked on but never snapped
home: its fixed hook is under the far lip, but its sprung foot rests on top of the near lip, so it stands a few
millimetres proud and tipped about 7.5 degrees. Neither the instruction nor the task file's docstring (which
language-model agents read) says which block it is: the robot has to find it, then press its near end down until the
sprung foot snaps under the lip, without unlatching or shifting its neighbours.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `tb_parts.py` | the rail and the jointed terminal block (A3C 2.5 outline, fixed hook and sprung foot), the USD writer, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted one-arm demo, closed loop on the simulated object poses |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the blocks start on the rail, four latched):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 600 \
        --instruction "One of the terminal blocks on the DIN rail is not latched. Find it and press it home so it latches, without unlatching or moving the other blocks."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 600-step trial is about 3.5 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant hold     # no arm moves: the faulty block stays as it started (score 0)
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right, with a zinc end bracket 4.2 cm in from each end; its centre 0.30 m in front of
  the arm bases and 0.04 m right of their midline.
- **Blocks:** five identical beige polyamide terminal blocks, 7.6 g, 5.1 mm thick (along the rail), 65 mm long
  (across it) and 36 mm tall (29 mm above the rail), side by side with 0.2 mm between them, the middle (third) one on
  the rail's centre. Each has a fixed hook at its far end (away from the arms) and a sprung foot at its near end.
- **The faulty block:** the fourth from the left. Its hook is under the far lip; its sprung foot's lead-in rests on
  the near lip's top edge. Its near end stands 4.9 mm above where a latched block's would, and it is tipped 7.6
  degrees (measured in the simulation after it settles; it stays there, untouched, for the whole run).

## Scoring

At the end of the episode, for the faulty (fourth) block:

| credit | condition |
| --- | --- |
| 1.0 | success: **latched** and still, held for 1 s in a row, with the other four blocks undisturbed |
| 0.5 | **seated**: tipped less than 3 degrees, but not latched |
| 0.25 | **lowered**: tipped less than 5 degrees |
| 0 | otherwise (including tipped further, or off the rail: its base 20 mm or more above the lips, or its centre outside the rail's footprint) |

- **Latched:** its base within -0.5..+0.8 mm of the lip tops at both lip edges, within 1 mm of the rail's centreline,
  its sprung foot in (within 1 mm of rest), both seat pads pressed by the lips, square and level within 3 degrees.
- **Undisturbed:** each of the other four blocks still latched and within 1 mm of where it started along the rail.
- If another block was disturbed, success is impossible and the score is capped at 0.25.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

Taking the faulty block off and mounting it again also scores 1.0: it ends latched, which is what a technician wants.

In a real trial these are read with a ruler, a feeler gauge and a photo: whether the block sits flat and square on
the lips (no gap under its near end, a gentle tug does not lift it), how far it is still tipped (the height of its
near end), and whether the other blocks have moved along the rail (marks on the rail) or come loose.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 600 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 126) | pressed from step 87 (the command leading the fingertips by up to 29 mm); lowered at step 103, seated at 105, latched at 107; other blocks moved at most 0.02 mm along the rail |
| CPU | **1.0** (success at step 126) | the same sequence |
| CPU, `--variant hold` | 0.0 (200 steps) | no arm moves: the faulty block stays tipped 7.6 degrees, the others latched |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 600
steps it lifted both arms and turned the wrists back and forth, but never came near the rail; the blocks stayed as
they started.

## The demo

The script knows which block is faulty (a policy has to see it). Right arm: fingers closed, comes down over the
faulty block's near end, 20 mm from its middle toward the arms, and presses down, the command leading (the arm is
soft), until the block latches; keeps pressing 0.5 s; lifts away. The closed fingertips also press the neighbouring
blocks' tops, which does no harm (they are latched). The left arm does not move.

## How it is built

yam_arena's task files only offer single primitives and library assets and have no extension point for jointed
parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified: `task.py` calls
`tb_parts.install()`, which wraps `yam_arena.scene.task_object` and `object_states` when that module is imported.
Parts are spawned from USD files written to a temporary folder.

- **Block:** an articulation. The housing's look is the A3C 2.5's outline, taken from its CAD at mid-thickness; its
  collision is two boxes. The sprung foot (near end) is a slider sprung toward the rail (1 N/mm), carrying the foot's
  tooth, lead-in and sole as convex pieces. The fixed hook (far end) has a 2.5 mm tooth (it can give up to 2.5 mm
  against a 1 N/mm spring, as in the dismount task built on the same parts; pressing does not use it). Two small
  sprung seat pads press on the lips, as the clamp of a latched block.
- **Faulty block:** spawned tipped about its hook tooth's inner corner under the far lip, by the angle that puts its
  sprung foot's lead-in on the near lip's top edge (computed from the parts' geometry; `TILT0` in `task.py`).
- **Rail:** fixed, 2 mm lips.

## Fidelity: what is simplified

- The latch is a spring stand-in: the sprung foot is a 1 N/mm slider, the seat pads 3 N/mm. Neither was measured.
- The block's collision shape is simplified (two boxes, plus convex pieces for the feet); its look is the CAD outline.
- The rail's lips are 2 mm thick (1 mm on a real rail): at 240 Hz a 1 mm lip lets parts tunnel through.
- Friction: block housing polyamide 0.35, latch parts on the lips 0.25, robot fingers PhysX default 0.5.
- The block's velocity (for "still") comes from its pose change: the latch joints' armature skews PhysX's.

## Known weak points

- From the fixed top camera the strip is a few pixels wide and the faulty block cannot be told apart; a wrist camera
  over the strip shows it clearly standing out of line. A policy has to go and look.
- The fault is always the fourth block; a policy that memorises the layout needs no inspection. Moving it is a
  one-line change (`FAULTY` in `task.py`) if reviewers want varied layouts.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- Object jitter must be 0: the blocks start on the rail.
- The scorer cannot see whether the robot is still touching the block when it counts success.
