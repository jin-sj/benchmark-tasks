# Open the knife disconnect on one named terminal block, leaving the others closed

Five identical knife-disconnect terminal blocks sit side by side on a DIN rail. Each has a small hinged lever on
top that opens or closes its circuit. The robot must open the lever of the second block from the left and leave the
other four closed. The levers are 4 mm wide on 6.2 mm-wide blocks, so the robot has to move one lever without
catching its neighbours.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `disconnect.py` | the strip of disconnect blocks: housings, hinged levers and their snap, the USD writer, the per-step state |
| `tb_parts.py` | the rail, shared shape and USD helpers, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted one-arm demo, closed loop on the simulated lever angles |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 600 \
        --instruction "Open the disconnect lever of the second terminal block from the left, and leave the other four levers closed."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 600-step trial is about 3.5 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant hold     # no arm moves: every lever stays closed (score 0)
    python demo.py --variant reclose  # opens the leftmost lever, closes it again, then the second: score 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right, with a zinc end bracket 4.2 cm in from each end; its centre 0.30 m in front of
  the arm bases and 0.04 m right of their midline.
- **Blocks:** five identical grey polyamide knife-disconnect terminal blocks, side by side with no gap, the middle
  (third) one on the rail's centre. Each is 6.2 mm thick (along the rail), 62 mm long (across it) and 34 mm tall
  above the rail's lips, latched on the rail.
- **Levers:** one per block, orange, 4 mm wide (along the rail), 18 mm long and 3 mm thick, hinged 2 mm beyond the
  block's middle (on the far side) about an axis along the rail, 1.5 mm above the block's top. Closed, a lever lies
  flat on the block's top with its free end toward the arms and a 6 mm tall, 3 mm thick upright tab at that end.
  Open, it stands upright, 95 degrees from closed, against a stop. It snaps: below 45 degrees a spring pulls it
  closed, above 45 degrees open (about 1 to 2 N at the tab to get it past the middle). All five start closed.

## Scoring

Interrupting a circuit that should stay live is the error this task is about, so touching another lever scores
zero, even if it is closed again. At the end of the episode:

| credit | condition |
| --- | --- |
| 0 | any of the other four levers was **lifted more than 10 degrees at any point** in the episode (even if closed again) |
| 1.0 | otherwise, the second lever from the left **open** and **kept open for the last 1 s** (success) |
| 0.5 | otherwise, the second lever was lifted **past its snap point** (45 degrees) at some point |
| 0.25 | otherwise, the second lever was lifted **at least 15 degrees** at some point |
| 0 | otherwise |

- **Open:** 80 degrees or more from its closed position (the stop is at 95).
- The task file records, every control step (20 Hz), each lever's largest angle so far and how long it has been open
  without a break; the score is read from these, so an episode cannot end in full credit after a neighbour was
  opened and closed again, nor with the second lever only just flipped open.

In a real trial these are read from a side-on video of the strip: the largest angle each lever reached (a protractor
card behind the strip, or the lever's tab against marks on the housing), and whether the second lever stood open for
the last second.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 600 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 180) | the lever pushed open at step 162 (the command leading the fingertip by up to 47 mm); full credit at step 180, once it had been open for 1 s; the other four levers never moved |
| CPU | **1.0** (success at step 182) | the score rose to 0.25 at step 142 (the lever at 16 degrees), 0.5 at step 161 (45 degrees) and 1.0 at step 182, once the lever had been open for 1 s; the other four levers never moved |
| GPU, `--variant reclose` | **0.0** (600 steps) | the leftmost lever opened at step 163 and pushed closed again at step 323; the second lever opened at step 522 and stayed open: the final state is the full demo's, but the score is 0 throughout |
| CPU, `--variant hold` | 0.0 (100 steps) | no arm moves: all five levers stay at 0 degrees |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`) on this version of the
task, scored 0.0: in 600 steps its left arm went to the rail's left end and its right arm stayed back; neither touched
a lever.

## The demo

Right arm: gripper open (47 mm between the pads). One fingertip (3.9 mm thick along the rail) comes down 6 mm short
of the second lever's tab, its end 4 mm below the tab's top, and pushes the tab away from the arms, the command
leading (the arm is soft), until the lever has snapped open; backs off 15 mm and lifts away. The other fingertip
stays beyond the strip's left end. The left arm does not move.

`--variant reclose` checks the scoring: the right arm first opens the leftmost lever, then pushes it closed again
(its fingertip against the upright lever's far side, pushed toward the arms), then opens the second lever as in the
demo. It ends in the state the full demo ends in, but scores 0: a neighbouring circuit was interrupted.

## How it is built

yam_arena's task files only offer single primitives and library assets and have no extension point for jointed
parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified: `task.py` calls
`tb_parts.install()`, which wraps `yam_arena.scene.task_object` and `object_states` when that module is imported.
Parts are spawned from USD files written to a temporary folder.

- **Strip:** one articulation with a fixed base (the five housings) and five hinged levers. A housing's collision
  is three boxes; a lever's is two (arm and tab).
- **Snap:** each hinge has a spring (0.02 N m/rad) whose target is 12 degrees beyond the stop the lever is nearer;
  once per control step (20 Hz) the target switches side when the lever crosses 45 degrees.
- **Rail:** fixed.

## Assets

No third-party assets: the rail, the disconnect blocks and their levers are this task's own geometry, built in code
(`disconnect.py`, `tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- The blocks are fixed on the rail (not latched parts that could come off): the task is about the lever.
- The snap is a spring whose target switches at 45 degrees, not an over-centre mechanism; its stiffness was not
  measured on a real block. The switch happens once per control step.
- The housing is a generic disconnect-terminal outline, not taken from a CAD model; the lever has no knife blade or
  electrical contact (open and closed are read from its angle).
- Friction: lever and housing polyamide 0.35, robot fingers PhysX default 0.5.

## Known weak points

- The lever is always the second from the left; a policy that memorises the layout need not count. Changing it is a
  one-line change (`TARGET` in `task.py`).
- From the fixed top camera the levers are a few pixels wide; a wrist camera over the strip shows them clearly.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- Holding the lever does not change the score: past 45 degrees it snaps open and its spring keeps it against the
  open stop (95 degrees) on its own, so a lever counted open for the last 1 s stays open when the robot lets go.
