# Take one terminal block off a loose DIN rail without disturbing its neighbours

Three terminal blocks are latched on a loose DIN rail. The robot must take the middle (blue) one off and put it in
a tray, leaving the two beige blocks on either side latched where they are, while keeping the rail itself in place
(one arm holds it down).

The block cannot be pulled straight off. Like a real push-in terminal block, its
base has a fixed hook at the far end and a sprung foot at the near end. To release it: push it about 2.5 mm toward
the arms (the fixed hook gives) so the sprung foot comes out from under its lip; tip the near end up a few degrees;
slide the block about 5 mm toward the far side so the hook comes out from under its lip; lift it off.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `tb_parts.py` | the rail, the tray and the jointed terminal block (this task's own generic push-in block: fixed hook and sprung foot), the USD writer, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted two-arm demo, closed loop on the simulated object poses |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the blocks start latched and must start exactly on the rail):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Take the blue terminal block off the DIN rail and put it in the tray. Hold the rail down with one arm while the other takes the block off, and leave the two beige blocks where they are."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 1,200-step trial is about 7 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant held     # the same, but the block is lowered onto the tray floor and never let go (0.75)
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks
    python demo.py --no-video

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, 200 g with its two zinc end brackets (4.2 cm in from each
  end), lying loose on the table, open side up, running left to right; its centre 0.30 m in front of the arm bases
  and 0.04 m right of their midline.
- **Blocks:** three terminal blocks latched on the rail, 33 mm apart centre to centre (28 mm gaps): beige, blue,
  beige from the left; the blue one 2 cm right of the rail's centre. Each is polyamide, 7.6 g, 5.1 mm thick (along
  the rail), 65 mm long (across it) and 36 mm tall (29 mm above the rail), the fixed hook at the far end (away from
  the arms) and the sprung foot at the near end.
- **Tray:** an open, shallow black plastic tray, inside 170 x 64 mm, walls 3 mm thick and 12 mm high, its centre
  0.18 m in front of the arm bases and 0.22 m right of their midline (in front of the right arm, nearer the arms than
  the rail), empty.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | success: the blue block **released and resting on the tray's floor**, still, held for 1 s in a row, with both beige blocks undisturbed and the rail in place |
| 0.75 | the blue block in the tray (its centre inside the tray's inner walls, its lowest point below their tops), released or still held |
| 0.5 | the blue block **off the rail**: its base 20 mm or more above the lips, or its centre outside the rail's footprint |
| 0.25 | the blue block **unlatched**: pushed more than 1 mm off-centre across the rail, its sprung foot or hook out more than 1 mm, its seat pads no longer pressed by the lips, its base lifted off the lips (more than 0.8 mm at either lip edge), or tipped or turned more than 3 degrees |
| 0 | otherwise |

- **Undisturbed:** each beige block still latched (base within -0.5..+0.8 mm of the lip tops at both lip edges,
  centred within 1 mm, feet in, square and level within 3 degrees) and within 1 mm of where it started along the rail.
- **Rail in place:** the rail's origin moved less than 10 mm across the table and it turned less than 3 degrees.
- If a beige block was disturbed or the rail was not in place, success is impossible and the score is capped at 0.25.
- **Released and resting on the tray's floor:** no part of the robot touching the block (the simulation reads the
  robot's contact force on the block's housing, which must stay under 0.05 N), and the block's lowest point within
  2 mm of the tray floor's top, so the tray, not the gripper, holds it up. Both must hold for the whole 1 s.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

In a real trial these are read with a ruler, a feeler gauge and a photo: whether the blue block lies in the tray with
the gripper open and clear of it (no contact), off
the rail or visibly shifted or tipped on it; whether the beige blocks have moved along the rail (marks on the rail)
or come loose; and how far the rail moved (marks on the table).

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,200 steps.

| physics | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 494) | pushed 2.5 mm toward the arms at step 241 (command 21 mm ahead of the arm), tipped 6 degrees by step 312, slid 5.0 mm past its start (hook out) by step 360, off the rail at step 390; let go just above the tray floor, the block resting on the floor with no robot contact from step 473; beige blocks undisturbed; rail moved 0.44 mm |
| CPU | **1.0** (success at step 517) | the same sequence |
| GPU, `--variant held` | **0.75** (610 steps) | the same until the tray; the block is lowered onto the tray floor and held there (the gripper presses it with about 24 N) to the end: in the tray, but never released, so no success |

The arm is soft in the simulation: the push and the slide each need the command to lead the gripper by 20-25 mm.

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`) on this version of the
task, scored 0.0: in 1,200 steps both arms moved over the rail and the tray, but the blue block stayed latched on the
rail.

## The demo

Left arm: fingers closed, pressed into the rail's channel 17 cm left of the rail's centre (between the left beige
block and the left end bracket), for the whole run.

Right arm: comes down over the blue block and grips it across its broad faces, 18 mm below its top. Pushes it toward
the arms, the command leading, until it has moved 2.4 mm (the fixed hook gives and the sprung foot's tooth comes out
past its lip's edge). Tips the near end up along an arc about the hook until the block is tipped 6 degrees. Slides it
toward the far side, the command leading, until it is 3.5 mm beyond where it started (the hook's tooth is out from
under its lip). Lifts it clear, carries it to the tray and lets go just above the floor; moves away.

`--variant held` checks the scoring: the same, but at the tray the right arm lowers the block onto the floor and keeps
holding it to the end of the run. The block is in the tray (0.75) but never released, so it cannot score 1.0.

## How it is built

yam_arena's task files only offer single primitives and library assets and have no extension point for jointed
parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified: `task.py` calls
`tb_parts.install()`, which wraps `yam_arena.scene.task_object` and `object_states` when that module is imported.
Parts are spawned from USD files written to a temporary folder.

- **Block:** an articulation. The housing's look is a generic push-in terminal block outline of this task's own design; its
  collision is two boxes. The sprung foot (near end) is a slider sprung toward the rail (1 N/mm), carrying the foot's
  tooth and sole as convex pieces. The fixed hook (far end) has a 2.5 mm tooth; it can slide outward by up to 2.5 mm
  against a 1 N/mm spring, which stands in for the plastic bending. Two small sprung seat pads press on the lips, as
  the clamp of a latched block.
- **Rail:** a free rigid body (steel, friction 0.3), 2 mm lips.

## Assets

No third-party assets: the rail, the tray and the terminal blocks are this task's own geometry, built in code
(`tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- The latch is a spring stand-in: the sprung foot is a 1 N/mm slider; the fixed hook's give is a 1 N/mm slide of up
  to 2.5 mm. That give was softened so the simulated arm, which pushes sideways weakly, can do the push; a real hook
  is stiffer. Neither stiffness was measured.
- The block's collision shape is simplified (two boxes, plus convex pieces for the feet); its look is a generic outline,
  not a particular product's.
- The rail's lips are 2 mm thick (1 mm on a real rail): at 240 Hz a 1 mm lip lets parts tunnel through.
- Friction: rail steel 0.3, block housing polyamide 0.35, latch parts on the lips 0.25, table and robot fingers
  PhysX default 0.5.
- The block's velocity (for "still") comes from its pose change: the latch joints' armature skews PhysX's.

## Known weak points

- The push needs a firm sideways force (about 2.5 N) from a gripper that holds the block only by friction; a policy
  that grips loosely slides along the block instead of moving it.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- Object jitter must be 0: the blocks start latched.
