# Price a case of cans with a hand price labeler

Stockers and order fillers (O*NET 53-7065.00; about 2.8 million US jobs, BLS) price merchandise as they stock it:
attaching price tags and labels to items is a listed task of the occupation. With a hand price labeler, each item
takes the same sequence: squeeze the labeler's trigger fully and let it go to feed one label out under its nose
roller, then press the roller onto the item where the label belongs. The robot must price a case of six soup cans:
one label on each lid, flat inside the lid's seam ring.

The physical skill: working a hand tool's trigger with the grip that holds the tool (squeezing it fully to feed a
label without letting the tool slip), then placing the tool's nose precisely on a small target and pressing it on,
six times, with one label per can: a second squeeze before the label is applied spoils it, and a label pressed onto
the seam ring or a can already priced is a mispriced can.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), the labeler's feed and the labels' sticking (once per control step), `success`, `score`, `distance`; docstring = what is on the table |
| `labeler_parts.py` | the case, the cans, the labeler (an articulation: its body and its sprung trigger), its stand and the labels |
| `tb_parts.py` | shared shape and USD helpers, and the hook that lets yam_arena spawn the parts and run the task's step hook |
| `demo.py` | scripted demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Price the cans: stick one price label flat on each can's lid, inside the lid's seam ring."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 2 control steps per second on one GPU with the
three cameras rendering (a full 1,200-step trial is about 10 minutes).

Scripted demo:

    python demo.py                      # GPU physics, as a trial runs: 1.0
    python demo.py --variant three      # three cans priced: 0.5
    python demo.py --variant double     # two labels on the first can, one can left unpriced: 0.5
    python demo.py --variant spoil      # one label spoiled by a second squeeze, all six cans priced: 0.5
    python demo.py --variant hold       # no arm moves: 0
    python demo.py --variant all        # all of them, one simulator session
    python demo.py --device cpu         # CPU physics, ~3x faster, for quick checks

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left.

- **Case**: an open cardboard tray, 148 x 218 mm, 3 mm floor, 40 mm walls, its middle 0.30 m in front of the arm
  bases and 0.12 m right of their midline.
- **Cans**: six steel soup cans standing upright, loose, in the case, two rows of three, 70 mm apart centre to
  centre: each 66 mm across, 100 mm tall, 0.42 kg, with a red printed wrap. On each lid, a seam ring around a flat
  centre panel 54 mm across.
- **Hand price labeler**: blue plastic, 0.2 kg, its handle 90 mm long and 15 mm thick with a rubber grip; in front of
  the handle (toward the trigger side, 10 mm away), a black trigger, sprung, squeezed 10 mm in to the handle (about
  7.5 N to start, 22.5 N fully in); at the handle's one end, the head (32 x 40 mm, 90 mm tall) reaching down to the
  nose roller (black rubber, 12 mm across), 65 mm along from the middle of the grip and 78 mm below it. Squeezed at
  least 8 mm in and let go, it feeds one label out under the roller. Its roll holds 20 labels.
- **Stand**: dark grey plastic, 70 x 110 mm, 5 mm base, with walls 62 mm tall round a slot 34 x 42 mm holding the
  labeler's head; it stands 0.22 m in front of the arm bases, 0.30 m right of their midline. The labeler stands in it nose
  down, its handle level along the table and its trigger facing away from the robot.
- **Labels**: 25 x 15 mm, white with an orange price band, self-adhesive.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | all six cans **priced** and no label anywhere else (success) |
| 0.5 | otherwise, at least three cans priced |
| 0.25 | otherwise, at least one can priced |
| 0 | otherwise |

- **Priced**: the can stands upright in the case (within 10 degrees of vertical) with **exactly one** label on its
  lid, the label **wholly inside the seam ring** (all four corners within the 54 mm centre panel).
- **Anywhere else**: a label stuck on anything but a can's lid (the case, the table, the stand), or **spoiled**: fed
  out while another label was already out (the first one is spoiled and discarded).
- A label sticks where the roller presses it, with the labeler held nose down (within 35 degrees of vertical), along
  the handle's direction; once on a lid it stays on that can (it is glued: it moves with the can).
- The task file keeps a record, every control step, of the labels fed, spoiled and stuck, and where; the score is
  read from it and from the cans' poses.

In a real trial: a photo of the case from above shows each lid's labels and whether each label lies inside the seam
ring; the labeler's roll counter (or the labels left) gives the number fed, so any label stuck elsewhere or spoiled is
accounted for.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,200 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 720) | the labeler taken out of its stand by step 136; six labels fed and pressed on, one per can, each inside the seam ring; no label elsewhere, none spoiled; the labeler carried back over its stand |
| CPU | **1.0** (success at step 727) | the same sequence |
| CPU, `--variant three` | **0.5** | three cans priced |
| CPU, `--variant double` | **0.5** | four cans priced: the first can has two labels, one can none |
| CPU, `--variant spoil` | **0.5** | all six cans priced, one label spoiled |
| CPU, `--variant hold` | 0 | no arm moves |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 1,200
steps its right arm moved about over the case and the stand but never took hold of the labeler.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
position), never the simulator's object poses.

The right arm, its gripper pointing down with its fingers closing across the labeler's handle and trigger, comes
down over the labeler in its stand, closes on the handle and trigger (pressing the trigger about 3 mm in: a firm hold,
short of feeding a label) and lifts it out. For each can in turn: it squeezes the trigger fully in and lets it go
(one label comes out), brings the nose roller 15 mm above the middle of the can's lid, lines up to within 1 mm,
presses the roller on 3 mm, and lifts it off.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Labeler**: an articulation, free: its body (handle, head and nose colliders) and its trigger on a prismatic joint
  with a preloaded spring (its drive).
- **Cans**: free rigid bodies; each collides as an eight-pointed section (a square and a cross of boxes) inside its
  round, flat-topped.
- **Labels**: free rigid bodies, held out of sight until the labeler feeds them (the labeler's roll). The task's step
  hook follows the trigger and the nose: a label pressed on is put where the roller pressed it, and on a lid it is
  then kept on its can every control step.

## Assets

No third-party assets: the case, the cans, the labeler, its stand and the labels are this task's own geometry, built
in code (`labeler_parts.py`); no meshes, scans or CAD files are used. The labeler is a generic design, not a copy of
any product.

## Fidelity: what is simplified

- The labels' adhesion is modelled by placing a pressed label where the roller pressed it and keeping it on its can;
  the label's peeling off the backing and its printing are not simulated.
- The labeler's feed mechanism is reduced to its trigger: a full squeeze and release feeds one label.
- The cans collide as eight-pointed sections inside their rounds.
- Friction: labeler grip and trigger rubber 0.9, other plastic 0.4, cans steel 0.3, case cardboard 0.5; robot
  fingers PhysX default 0.5.
