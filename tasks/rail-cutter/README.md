# Cut a DIN rail to length on a powered rail cutter

Electricians and panel assemblers cut DIN rail to length for every control panel they build. On a powered bench
rail cutter: slide the rail through the cutter's die until its end reaches the right mark on the length scale,
clamp it, and press the cut button; the cutter shears it. The robot must do the same: cut a 250 mm length off the
right end of a 600 mm rail.

The physical skill: two arms on one workpiece and one machine. One arm feeds a long, light rail through a
close-fitting die and sets its end at a mark on the scale to within 1 mm, reading the scale, and keeps it there;
the other shuts the toggle clamp on the rail without disturbing it and presses the cut button. The length counts
only if the rail did not move during the cut.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `cutter.py` | the rail cutter: its base (ledges, guide fences, die, scale), cutting head and blade, toggle clamp, cut button and interlock |
| `tb_parts.py` | the rail, shared shape and USD helpers, and the hook that lets yam_arena spawn the parts and run the record every control step |
| `demo.py` | scripted two-arm demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Cut a 250 mm length off the right end of the rail with the rail cutter."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 1,200-step trial is about 7 minutes).

Scripted demo:

    python demo.py                      # GPU physics, as a trial runs: 1.0
    python demo.py --variant short      # the rail is set 3 mm short (247 mm) before the cut: 0.5
    python demo.py --variant noclamp    # the button is pressed with the clamp open: no cut (interlock): 0.25
    python demo.py --variant nocut      # the rail is set at 250 mm and clamped, the button never pressed: 0.25
    python demo.py --variant hold       # no arm moves: 0
    python demo.py --variant all        # all of them, one simulator session
    python demo.py --device cpu         # CPU physics, ~3x faster, for quick checks

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left.

- **Powered bench DIN-rail cutter**, fixed to the table, its blade's plane 0.04 m right of the midline, its rail path
  0.32 m in front of the arm bases, running left to right. Blue steel body:
  - **Infeed ledge** (left of the blade): 60 mm wide, 40 mm tall, from 12 mm to 680 mm left of the blade. Along it,
    two **guide fences**, 3 mm thick and 4 mm tall, their inner faces 14 mm either side of the rail path, guiding the
    rail's 27 mm base (0.5 mm of play each side) under its lips.
  - **Die**, around the blade: 24 mm long along the rail, its opening 37 mm wide and 9.5 mm tall above the ledge's
    height (the rail is 35 x 7.5 mm), its mouth on the infeed side flared over 4.5 mm (3 mm wider and taller at the
    face).
  - **Cutting head** over the die: a dark grey housing 64 x 44 mm, 110 mm tall, with a yellow guard band round its
    foot; the blade (steel, 1.5 mm thick) runs inside it, enclosed. Rated shearing force 30 kN (a TS35 x 7.5 rail of
    1 mm steel shears at about 15 kN); one stroke: down in 1.0 s, back up in 0.5 s.
  - **Outfeed ledge** (right of the blade): 60 mm wide, 40 mm tall, from 12 mm to 330 mm right of the blade. On its
    top, on the side nearer the arms, a **length scale** measured from the blade's plane: black marks every 10 mm
    (4 mm long), every 50 mm (7 mm) and every 100 mm (10 mm), to 300 mm. No length stop is fitted.
  - **Toggle hold-down clamp** on the infeed side, 45 mm from the blade: a steel arm hinged on a post behind the
    rail (40 mm behind the rail path, 80 mm above the table), 205 mm long, pointing toward the arms, with a red rubber
    handle (80 mm long, 22 mm square) at its end and a black rubber pad (40 x 30 mm) over the rail near the hinge. It
    rests open, raised 45 degrees. Pushed down past 20 degrees it snaps shut (over centre), the pad pressing the rail
    onto the ledge (about 7 N); lifted back past 20 degrees it springs open. About 7 N at the handle moves it.
  - **Console** at the front, on the outfeed side (65 to 105 mm in front of the rail path, 35 to 75 mm right of the blade),
    60 mm tall, with a red **cut button** on top: a 20 mm square cap standing 6 mm proud, pressed 3 mm in (about
    5 N) it starts one stroke, if the clamp is shut. With the clamp open the button does nothing (the interlock); the
    button must come back up before it starts another stroke.
- **Rail**: a loose steel TS35 x 7.5 top-hat DIN rail, 600 mm long, 0.2 kg, open side up, lying on the infeed ledge
  between the fences, its right end 40 mm short of the blade.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | the **cut** was made with the rail in the die and its right end **250 mm past the blade, within 1 mm**, and the rail **moved less than 0.5 mm** along the cutter during the stroke (success) |
| 0.5 | otherwise, the cut was made with the rail in the die and its end within **5 mm** of 250 mm past the blade |
| 0.25 | otherwise, there was no such cut, but before the cut the rail lay in the die with its end within 5 mm of the 250 mm mark **for 1 s without a break** (it was set at the mark) |
| 0 | otherwise |

- **The cut** is the first stroke of the blade, made at the bottom of the stroke. Only the first stroke counts; a
  rail cut wrong cannot be cut again. A stroke with no rail in the die is a cut too (it scores 0). The cutter makes a
  stroke only with the clamp shut.
- **During the stroke**: from the button's starting it to the blade's reaching the bottom.
- **In the die**: the rail runs along the cutter (within 3 degrees), lies on the die's floor (within 2 mm), is
  centred in the opening (within 2 mm) and crosses the blade's plane.
- The length is measured from the blade's plane to the rail's end on the outfeed side, whichever way round the rail
  lies.
- The task file keeps a record, every control step (20 Hz), of the rail's end, whether the rail is in the die, the
  stroke, and the cut; the score is read from it.

In a real trial: the cut piece is measured with a ruler (250 mm, within 1 mm: full credit; within 5 mm: half), and a
video of the cutter shows whether the rail moved against the scale's marks during the stroke and, with no cut,
whether the rail was set at the mark for a second.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,200 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 729) | the rail set by step 309; the clamp shut by step 561; the button pressed at step 723; cut 249.5 mm, the rail moved 0.00 mm during the stroke |
| CPU | **1.0** (success at step 703) | the same sequence; cut 251.0 mm, the rail moved 0.00 mm |
| CPU, `--variant short` | **0.5** | cut 247.9 mm |
| CPU, `--variant noclamp` | **0.25** | the button pressed with the clamp open: no stroke; the rail set at 250 mm |
| CPU, `--variant nocut` | **0.25** | the rail set at 250 mm and clamped, no cut |
| CPU, `--variant hold` | 0 | no arm moves |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 1,200
steps its arms moved over the two ends of the cutter (the rail on the infeed side, the scale on the outfeed side) but
never gripped the rail, the clamp or the button.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
positions), never the simulator's object poses.

- **Left arm** (the rail): the gripper pointing down, its fingers closing across the rail, comes down over the rail
  100 mm from its left end, closes on the rail's lips (clear of the fences), slides the rail 290 mm toward the outfeed
  over 4 s, which puts its right end 250 mm past the blade, lines its measured gripper position up with that point to
  within 0.3 mm, and holds it there.
- **Right arm** (the clamp and the button), once the rail is set: comes down over the clamp's handle, closes on it
  across its sides, pushes it down along its arc (its command leading the handle downward against the spring) until
  the gripper is where the handle is with the clamp 3 degrees from level, lets go (the clamp stays shut), then, its
  gripper closed and pointing down, presses the cut button with its fingertips and lifts off.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Cutter**: an articulation with a fixed base (boxes: ledges, fences, die, head, console, scale marks, post) and
  three joints: the clamp (revolute, 0 to 45 degrees, its over-centre spring a joint drive whose target switches at 20
  degrees), the button (prismatic, 5 mm of travel, sprung up) and the blade (prismatic, its 50 mm stroke driven by the
  cutter's cycle, once per control step).
- **Rail**: a free rigid body; its collision shape is the top-hat profile (base, walls and lips as boxes).
- **Record**: `task._record` runs once per control step, from the rail's pose and the cutter's joints.

## Assets

No third-party assets: the rail and the cutter are this task's own geometry, built in code (`cutter.py`,
`tb_parts.py`); no meshes, scans or CAD files are used. The cutter is a generic design, not a copy of any product.

## Fidelity: what is simplified

- The blade is a look only and the rail is not divided: the cut is the stroke reaching the bottom, and the cut length
  is the rail's position at that moment.
- The cutter's motor and the cutting force are not simulated: the blade follows its stroke.
- Friction: rail and cutter steel 0.3; clamp pad and handle rubber 0.9; robot fingers PhysX default 0.5.
