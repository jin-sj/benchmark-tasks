# Put a spare fuse into the storage pocket of one fuse terminal block in a mixed strip

Fuse terminal blocks often keep a spare fuse in a small pocket next to the fuse holder. Here a mixed strip of nine
terminal blocks is latched on a DIN rail; three of them, in the middle, are fuse blocks with an empty spare-fuse
pocket each. One 5 x 20 mm glass cartridge fuse stands in a rack near the robot. The robot must pick it up and store
it in the pocket of the middle fuse block: a 5 mm fuse into a 6 mm hole, beside the block's upright fuse holder, in
the right one of three look-alike blocks.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `fuse_strip.py` | the mixed strip (plain blocks, fuse blocks with their pockets and holder plates) and the fuse rack |
| `tb_parts.py` | the rail, the fuse, shared shape and USD helpers, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted one-arm demo, closed loop on the simulated object poses |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the fuse starts standing in the rack):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 800 \
        --instruction "Take the spare fuse from the rack and put it into the spare-fuse pocket of the middle one of the three fuse terminal blocks on the DIN rail."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 5 control steps per second on one GPU with the
three cameras rendering (a full 800-step trial is about 3 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant hold     # no arm moves: the fuse stays in the rack (score 0)
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right, with a zinc end bracket 4.2 cm in from each end; its centre 0.30 m in front of
  the arm bases and 0.04 m right of their midline.
- **Strip:** nine polyamide terminal blocks latched side by side, no gaps; from the left: two blue plain blocks
  (5.1 mm thick along the rail), one grey plain block (6.2 mm), three grey fuse blocks (8.2 mm each), two beige plain
  blocks (5.1 mm), one grey plain block (6.2 mm). The middle fuse block is on the rail's centre. A plain block is
  52 mm long (across the rail) and 28 mm tall above the rail's lips; a fuse block is 62 mm long and 34 mm tall.
- **Fuse blocks:** each carries an upright fuse-holder plate (25 mm long, 24 mm tall, 6 mm thick) on the far half of
  its top, starting 5 mm toward the arms from the rail's centreline, and a spare-fuse pocket in the near half of its
  top: a square hole 6 mm wide and 14 mm deep, its axis 14.5 mm toward the arms from the rail's centreline, with a
  45 degree lead-in at its mouth (2 mm on its near and far edges, 0.9 mm on its side edges). All three pockets are
  empty.
- **Rack:** black plastic, 30 x 30 x 12 mm, fixed to the table, its centre 0.18 m in front of the arm bases and
  0.22 m right of their midline, with a 6.6 mm square hole 8 mm deep in the middle of its top.
- **Fuse:** one 5 x 20 mm glass cartridge fuse (1 g), standing upright in the rack's hole, 12 mm of it out of the
  rack.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | success: the fuse **stored** in the middle fuse block's pocket and still, held for 1 s in a row |
| 0.5 | the fuse **partly in** that pocket |
| 0.25 | the fuse **taken from the rack**, wherever it is (on the table, in the gripper, in another block's pocket) |
| 0 | otherwise |

- **In the pocket:** the fuse's centre within 1.5 mm of the pocket's axis, and its lower end 2 mm or more below the
  pocket's mouth (the block's top).
- **Stored:** in the pocket, its lower end within 1 mm of the pocket's floor (so 6 mm of the fuse stands out of the
  block's top; partly in: more than 7 mm).
- **Taken from the rack:** the fuse's centre 30 mm or more from where it started.
- **Still:** under 10 mm/s and 0.5 rad/s.

In a real trial these are read with a ruler and a photo: which pocket the fuse is in, and how far it stands out of
the block's top (6 mm when stored).

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 800 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 473) | fuse gripped at step 160 (gripper within 0.5 mm of its target), out of the rack by step 190; its lower end lined up within 0.3 mm of the pocket's axis; lowered until step 420, 3.6 mm above the floor when let go; it drops onto the floor |
| CPU | **1.0** (success at step 467) | the same sequence |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 800
steps its right arm went over the strip, not to the rack, and the fuse stayed in the rack.

## The demo

Right arm, the gripper turned so its pads close across the rail's direction (the fingers' wide side then runs along
the rail, clear of the blocks' fuse-holder plates). It comes down over the fuse in the rack in three stages, each
waiting for the measured gripper position to settle, grips the fuse 8.5 mm below its upper end and lifts it out. It
carries the fuse over the middle fuse block's pocket, 12 mm above it, and moves until the fuse's lower end is within
0.6 mm of the pocket's axis. It lowers the fuse at 4 mm/s, still lining it up until it has entered, until the pad
tips are 1.5 mm above the block's top, lets go (the fuse drops the last few millimetres) and lifts away. The left arm
does not move.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Strip and rack:** fixed colliders made of boxes, with small wedges for the pockets' lead-ins.
- **Fuse:** a free rigid body. It looks round (a glass tube between two metal caps); its collision shape is a 5 mm
  square bar.

## Fidelity: what is simplified

- The pocket and the rack are stand-ins, not taken from a manufacturer's CAD model: the pocket's size (6 mm square,
  14 mm deep), its lead-in and how far a stored fuse stands out (6 mm) are this task's choices. On real blocks the
  pocket holds the fuse with light friction; here the fit is loose and the fuse rests on the floor.
- The blocks are fixed on the rail (not latched parts that could come off), and their housings are generic outlines.
  The fuse-holder plate is a plain plate: it does not open and holds no fuse.
- The spare fuse is presented standing in a rack. A fuse lying loose in a box cannot be picked up and stood upright
  by this gripper.
- The fuse's collision shape is a square bar (a cylinder collider slipped out of the simulated fingertips); it does
  not roll, and the glass cannot break.
- Friction: fuse 0.4, housings polyamide 0.35, rack plastic 0.4, robot fingers PhysX default 0.5.

## Known weak points

- The target is always the middle fuse block; a policy that memorises the layout need not tell the blocks apart.
  Changing it is a one-line change (`TARGET` in `task.py`).
- From the fixed top camera the strip is small and the pockets are a few pixels wide; a wrist camera over the strip
  shows them.
- The simulated arm wanders by about 0.7 mm when holding still, which is why the pocket has a generous lead-in.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- The scorer cannot see whether the robot is still holding the fuse when it counts success.
