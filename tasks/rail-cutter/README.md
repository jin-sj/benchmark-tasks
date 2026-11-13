# Cut a DIN rail to length with a bench rail cutter

Electricians and panel assemblers cut DIN rail to length for every control panel they build, on a bench rail
cutter: slide the rail through the cutter's die until its end reaches the right mark on the length scale, hold it
there, and pull the lever down to shear it. The robot must do the same: cut a 250 mm length off the right end of a
600 mm rail.

The physical skill: two arms in sequence on one workpiece. One arm feeds a long, light rail through a close-fitting
die and sets its end at a mark to within 1 mm, then keeps holding it still; the other pulls a sprung lever through
its stroke along its arc. The length counts only if the rail is not moved while the blade is in it.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `cutter.py` | the rail cutter: its fixed base (ledges, guide fences, die, scale) and its sprung, hinged lever |
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

    python demo.py                    # GPU physics, as a trial runs: 1.0
    python demo.py --variant short    # the rail is set 3 mm short (247 mm) before the cut: 0.5
    python demo.py --variant wiggle   # the rail is pushed on and back while the blade is in it: 0.5
    python demo.py --variant nocut    # the rail is set at 250 mm, the lever is never pulled: 0.25
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --variant all      # all of them, one simulator session
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left.

- **Rail cutter**, fixed to the table, its blade's plane 0.04 m right of the midline, its rail path 0.32 m in front of
  the arm bases, running left to right. Blue cast steel body:
  - **Infeed ledge** (left of the blade): 60 mm wide, 40 mm tall, from 12 mm to 680 mm left of the blade. Along it,
    two **guide fences**, 3 mm thick and 4 mm tall, their inner faces 14 mm either side of the rail path: they keep
    the rail's 27 mm base straight (0.5 mm of play each side) under its lips.
  - **Die**, around the blade: 24 mm long along the rail, its opening 37 mm wide and 9.5 mm tall above the ledge's
    height (the rail is 35 x 7.5 mm), its mouth on the infeed side flared over 4.5 mm in three 1.5 mm steps (each
    side and the top 3, 2 and 1 mm further out).
  - **Outfeed ledge** (right of the blade): 60 mm wide, 40 mm tall, from 12 mm to 330 mm right of the blade. On its
    top, on the side nearer the arms, a **length scale** measured from the blade's plane: black marks every 10 mm
    (4 mm long), every 50 mm (7 mm) and every 100 mm (10 mm), to 300 mm.
  - **Lever**: hinged on a post behind the die, about an axis along the rail, 40 mm behind the rail path and 80 mm
    above the table. A steel arm 205 mm long, 16 x 12 mm, points toward the arms, ending in a red rubber handle 80 mm
    long, 22 mm square; the blade (steel, 1.5 mm thick) hangs from the arm by the die. A spring holds the lever at its
    upper stop, 45 degrees up; pulling the handle down until the lever is level (about 10 N at the handle) drives the
    blade through the die.
- **Rail**: a loose steel TS35 x 7.5 top-hat DIN rail, 600 mm long, 0.2 kg, open side up, lying on the infeed ledge
  between the fences, its right end 40 mm short of the blade.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | the **cut** was made with the rail in the die and the rail's right end **250 mm past the blade, within 1 mm**, and the rail **moved less than 0.5 mm** along the cutter while the blade was in it (success) |
| 0.5 | otherwise, the cut was made with the rail in the die and its end within **5 mm** of 250 mm past the blade |
| 0.25 | otherwise, there was no such cut, but before the cut the rail lay in the die with its end within 5 mm of the 250 mm mark **for 1 s without a break** (it was set at the mark) |
| 0 | otherwise |

- **The cut** is the lever's first full stroke: the first time the lever comes within 4 degrees of level. Only the
  first stroke counts; a rail cut wrong cannot be cut again. A stroke with no rail in the die is a cut too (it
  scores 0).
- **In the die**: the rail runs along the cutter (within 3 degrees), lies on the die's floor (within 2 mm), is
  centred in the opening (within 2 mm) and crosses the blade's plane.
- **While the blade was in it**: from the lever's reaching 20 degrees (where the blade meets the rail's top) to the
  cut, within one stroke.
- The length is measured from the blade's plane to the rail's end on the outfeed side, whichever way round the rail
  lies.
- The task file keeps a record, every control step (20 Hz), of the rail's end, whether the rail is in the die, the
  lever's angle, and the cut; the score is read from it.

In a real trial: the cut piece is measured with a tape measure or a ruler (250 mm, within 1 mm: full credit; within
5 mm: half), and a video of the cutter shows whether the rail moved while the lever came down (against the scale's
marks), and, with no cut, whether the rail was set at the mark for a second. A real cutter's cut is clean and square;
any burr is ignored.

## Measured results

Scripted demo, 20 Hz control, object jitter 0.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 568) | the rail set by step 309; cut 249.4 mm, the rail moved 0.17 mm during the stroke |
| CPU | **1.0** (success at step 573) | the rail set by step 310; cut 250.3 mm, the rail moved 0.02 mm |
| CPU, `--variant short` | **0.5** | cut 247.0 mm |
| CPU, `--variant wiggle` | **0.5** | the rail moved 2.7 mm under the blade; cut 252.8 mm |
| CPU, `--variant nocut` | **0.25** (800 steps) | the rail held at 250 mm, no cut |
| CPU, `--variant hold` | 0 (800 steps) | no arm moves |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in
1,200 steps its arms hovered over the two ends of the cutter, the left over the rail on the infeed ledge and the
right over the outfeed's scale, but neither gripped the rail or the lever.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
positions), never the simulator's object poses.

- **Left arm** (the rail): the gripper pointing down, its fingers closing across the rail (the wrist turned the way
  that keeps it in range), comes down over the rail 100 mm from its left end, closes on the rail's lips (the pad tips
  1.8 mm below the lips' tops, clear of the fences), slides the rail 290 mm toward the outfeed over 4 s, which puts
  its right end 250 mm past the blade, lines its measured gripper position up with that point to within 0.3 mm, and
  holds it there.
- **Right arm** (the lever), once the rail is set: comes down over the lever's handle, closes on it across its
  sides, then pulls it down along its arc at 15 degrees a second (its command leading the handle downward, since the
  spring pushes back) until the gripper is where the handle is with the lever level, holds a moment, lets the lever
  rise again along its arc, and lets go.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Cutter**: an articulation with a fixed base (boxes: ledges, fences, die, scale marks, post) and one revolute joint
  for the lever, with a spring (a joint drive toward 60 degrees, beyond the 45 degree stop) and limits 0 to 45
  degrees.
- **Rail**: a free rigid body; its collision shape is the top-hat profile (base, walls and lips as boxes).
- **Record**: `task._record` runs once per control step, from the rail's pose and the lever's joint angle.

## Assets

No third-party assets: the rail and the cutter are this task's own geometry, built in code (`cutter.py`,
`tb_parts.py`); no meshes, scans or CAD files are used. The cutter is a generic design, not a copy of any product.

## Fidelity: what is simplified

- Nothing is cut: the blade is a look only (it does not collide) and the rail stays whole. The cut is the lever's
  stroke, and the cut length is the rail's position at that moment.
- The lever is soft: about 10 N at the handle holds it down, and nothing resists the blade entering the rail. A real
  cutter needs a much stronger pull (its lever is longer, or the shear has a ratchet), and the rail pushes back
  against the blade.
- A real cutter has an adjustable length stop on its outfeed; this one has a scale only, so the rail must be set by
  eye or by measurement.
- The die's opening is 1 mm wider than the rail each side; the guide fences keep the rail within 0.5 mm.
- Friction: rail and cutter steel 0.3; lever handle rubber 0.9; robot fingers PhysX default 0.5.

## Known weak points

- The length is always 250 mm and the rail always starts at the same place: a policy that memorises one slide of
  290 mm need not read the scale.
- From the fixed top camera the scale's marks are a pixel or two apart; the wrist cameras show them.
- The blade's not colliding means a policy could, in principle, slide the rail under a lowered lever; the first
  stroke is the cut, so lowering the lever with no rail in the die scores 0.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
