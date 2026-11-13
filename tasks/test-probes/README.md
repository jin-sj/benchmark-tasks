# Measure the voltage between two named terminals with a pair of test probes

A control panel's terminal strip has 40 terminal blocks side by side; each has a small test opening on top that a
multimeter probe tip fits into. The robot must pick up the two test probes, one per arm, and hold one tip in the test
opening of the grey line (L) terminal and the other in the blue neutral (N) terminal, both at once, for a second, as
an electrician does to check a circuit's voltage. A tip put into any other terminal's opening fails the task: on a
live panel it measures the wrong circuit, and probing around a strip is how faults and shorts are caused.

The physical skill: two arms each placing a thin tool's 2 mm tip into a 2.6 mm opening 19 cm from the other, and
holding both steady together, without touching any of the 38 other openings.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `probe_parts.py` | the terminal strip with its test openings, the two probes and their holders, and the per-step record the scorer reads |
| `tb_parts.py` | the rail, shared shape and USD helpers, and the hook that lets yam_arena spawn the parts and run the record every control step |
| `demo.py` | scripted two-arm demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the probes start standing in their holders):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Measure the voltage between the L terminal and the N terminal: hold one probe's tip in each terminal's test opening, both at once, for one second. Do not put a probe tip into any other terminal's opening."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 1,200-step trial is about 7 minutes).

Scripted demo:

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant wrong    # the black probe goes into a terminal two blocks from N first, then into N: 0
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right; its centre 0.30 m in front of the arm bases, on their midline.
- **Terminal strip:** 40 polyamide terminal blocks latched side by side at the rail's middle, 5.2 mm apart centre to
  centre, each 56 mm long across the rail and 40 mm tall above its lips. From the left: a green-yellow earth block,
  the **grey L block**, 35 beige blocks (other circuits), the **blue N block**, a green-yellow earth block and a
  beige block. L is 96.2 mm left of the rail's centre, N 91.0 mm right of it: 187 mm apart.
- **Test openings:** on top of every block at its middle (on the rail's centreline): a square hole 2.6 mm wide and
  8 mm deep, with a 45 degree lead-in (a funnel) at its mouth (1.5 mm on its near and far edges, 1.0 mm on its
  sides), its floor the block's metal clamp.
- **Probe holders:** two black plastic blocks, 30 x 30 x 30 mm, fixed to the table 0.20 m in front of the arm bases,
  0.14 m left and right of their midline, each with an 8 mm square hole in its top.
- **Probes:** a red one in the left holder and a black one in the right, standing tip down. Each is 140 mm long,
  30 g: a 2 mm steel tip 15 mm long, a 6 mm shroud 10 mm long (in the holder's hole), a 16 mm finger guard resting
  on the holder's top, a 10 mm handle 100 mm long and a 6 mm cap. Their leads are left out (see fidelity).

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 0 | a probe tip **entered any other terminal's test opening** at any point (1 mm or more below its mouth, inside its bore) |
| 1.0 | otherwise, **measured**: both tips seated in the L and N openings (one in each, 3 mm or more below the mouth, inside the bore) and both probes held, without a break for 1 s (success) |
| 0.5 | otherwise, one tip was seated in the L or N opening while its probe was held, at some point |
| 0.25 | otherwise, both probes were lifted 30 mm or more out of their holders at some point |
| 0 | otherwise |

- **Held:** a closed gripper on the probe's handle: the gripper's centre within 4 mm of the probe's axis, along the
  handle, its pads closed to the handle's width plus 1.5 mm at most. (A probe left standing in an opening on its own
  does not count: in a real trial the probe is in the robot's hand.)
- Seated means the tip is down on the clamp, where a real probe makes contact; it is read from the probe's pose.
- The task file keeps a record, every control step (20 Hz), of each tip's position, which opening it is in and how
  deep, whether its probe is held, how far it was lifted, any entry into a wrong opening, and how long both tips
  have been seated and held without a break; the score is read from it.

In a real trial these are read from a video of the strip (top and side) and the robot's grippers: which opening each
tip went into and when, and whether both stayed in for a second with the probes in the grippers. A multimeter on the
probes' leads, with the L and N terminals at a test voltage, confirms the measurement (it reads only with both tips on
their clamps).

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,200 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 562) | both probes lifted out by step 202; the red tip went into L at the first try (seated at step 349); the black tip found N at its third try (the first two landed on the block's top beside the opening); both seated and held from step 546, success 1 s later; no tip entered any other opening |
| CPU | **1.0** (success at step 379) | the same sequence, both tips in at the first try |
| GPU, `--variant wrong` | **0** (800 steps) | the black tip entered the opening of the beige block two from N at step 544 (wrong opening): 0 from then on |
| CPU, `--variant wrong` | **0** (800 steps) | the same, at step 316; it then went on into N, with the red tip in L: the demo's final state, still 0 |
| CPU, `--variant hold` | 0.0 (100 steps) | no arm moves |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 1,200
steps its arms moved about over the table, but neither picked up a probe.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
position), never the simulator's object poses. Each arm, the gripper pointing down with its fingers closing across
the rail (each arm turned the way that keeps its last wrist joint in range): comes down over its probe's handle,
closes on it 80 mm above the tip (near its centre of mass, so it does not pivot in the grip), lifts it 50 mm out of
its holder, carries it over its terminal's test opening, lines its measured gripper position up over the opening to
within 0.4 mm, and lowers the tip at 6 mm/s, its command never more than 3 mm ahead of the gripper, until the tip is
5 mm below the opening's mouth. It feels for the opening: if the gripper stops descending early (the tip is on the
block's top, beside the opening), it lifts the tip 4 mm, shifts 1 or 2 mm and tries again, up to nine times. Then it
holds. The left arm takes the red probe to L, the right arm the black probe to N.

`--variant wrong` checks the scoring: the right arm first puts the black probe into a beige terminal two blocks from
N, takes it out, then puts it into N as in the demo. It scores 0 from the moment the tip enters the wrong opening.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Strip and holders:** fixed colliders made of boxes, with small wedges for the openings' lead-ins.
- **Probes:** free rigid bodies; their collision shapes are square sections (tip, shroud, guard, handle, cap).
- **Record:** `probe_parts._record` runs once per control step, from the probes' poses and each arm's joint readings
  (its gripper centre from the arm's forward kinematics, and its finger gap).

## Assets

No third-party assets: the rail, the terminal strip, the holders and the probes are this task's own geometry, built
in code (`probe_parts.py`, `tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- The probes have no leads. Real probes trail about 1 m of cable each to the meter, which pulls on them a little;
  here they are free.
- No electricity is simulated: a tip counts as making contact when it is seated on the opening's floor (the clamp).
- The terminal blocks are fixed colliders of a generic design (not latched parts that could come off); the test
  openings are square, real ones are usually round.
- The probe's collision shapes are square sections (a round one slipped in the simulated gripper).
- Friction: probe handle rubber 0.9, other plastic 0.4, housings polyamide 0.35, robot fingers PhysX default 0.5.

## Known weak points

- L and N are always the same two blocks; a policy that memorises the layout need not tell the blocks apart.
- The two terminals are 187 mm apart: closer than about 17 cm, the two arms' wrists touch and push each other off
  their openings.
- From the fixed top camera the openings are a pixel or two wide; the wrist cameras show them.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
