# Measure the voltage between two named terminals with a multimeter and its test probes

A control panel's terminal strip has 40 terminal blocks side by side; each has a small test opening on top that a
multimeter probe tip fits into, down to the block's metal clamp. A 12 V DC source is wired into the grey line (L)
and blue neutral (N) terminals. A handheld multimeter lies switched off in front of the strip, its probes standing in
two holders. The robot must take a measurement the way an electrician does: switch the meter to DC volts, then hold
one probe's tip on the L terminal's clamp and the other on the N terminal's clamp, both at once, until the meter
shows the voltage steadily. The score is read from the meter's display. A tip put into any other terminal's
opening fails the task: on a live panel it measures the wrong circuit, and probing around a strip is how faults and
shorts are caused.

The physical skill: turning the meter's clicking rotary selector to the right position, then two arms each placing a
thin tool's 2 mm tip into a 2.6 mm opening 19 cm from the other and down onto its clamp, and holding both steady on
their clamps together, without touching any of the 38 other openings.

**Occupations** (taking voltage readings with a multimeter and test probes is a listed, regular task):
electricians (O\*NET 47-2111.00, task "Test electrical systems or continuity of circuits in electrical wiring,
equipment, or fixtures, using testing devices such as ohmmeters, voltmeters, or oscilloscopes, to ensure
compatibility and safety of system"; BLS: about 0.8 million US workers), and industrial machinery mechanics
(O\*NET 49-9041.00, who test machines' electrical controls with test meters when diagnosing faults; BLS: about 0.4
million). Checking for voltage at a panel's terminals with a two-probe meter is everyday work for both.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `probe_parts.py` | the terminal strip with its test openings, the two probes and their holders, the circuit, and the per-step record (with the meter's readings) the scorer reads |
| `meter_parts.py` | the multimeter: its case, its clicking rotary selector (a jointed knob) and its seven-segment display |
| `tb_parts.py` | the rail, shared shape and USD helpers, and the hook that lets yam_arena spawn the parts and run the record every control step |
| `demo.py` | scripted two-arm demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the probes start standing in their holders):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1600 \
        --instruction "Measure the DC voltage between the L terminal and the N terminal with the multimeter: the meter must show it steadily for at least one second while one probe's tip is held in each terminal's test opening. Do not put a probe tip into any other terminal's opening."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 1,600-step trial is about 9 minutes).

Scripted demo:

    python demo.py                    # GPU physics, as a trial runs: 1.0
    python demo.py --variant brief    # both tips on their clamps for only ~0.8 s: 0.5
    python demo.py --variant nodial   # the meter is left OFF; both tips go onto their clamps: 0.25
    python demo.py --variant wrong    # the black tip goes into a terminal two blocks from N first, then into N: 0
    python demo.py --variant hold     # no arm moves: 0
    python demo.py --device cpu       # CPU physics, ~3x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right; its centre 0.30 m in front of the arm bases, on their midline.
- **Terminal strip:** 40 polyamide feed-through terminal blocks (screw clamp, 2.5 mm² class) latched side by side at
  the rail's middle, 5.2 mm apart centre to centre, each 56 mm long across the rail and 40 mm tall above its lips.
  From the left: a green-yellow earth block, the **grey L block**, 35 beige blocks (other circuits), the **blue N
  block**, a green-yellow earth block and a beige block. L is 96.2 mm left of the rail's centre, N 91.0 mm right of
  it: 187 mm apart.
- **Test openings:** on top of every block at its middle (on the rail's centreline): a square hole 2.6 mm wide and
  8 mm deep, with a 45 degree lead-in (a funnel) at its mouth (1.5 mm on its near and far edges, 1.0 mm on its
  sides), its floor the block's metal clamp.
- **Test voltage:** an isolated 12 V DC source wired into the clamps of the L block (+) and the N block (-) through
  their wire entries on the far side; no other block is wired.
- **Multimeter:** a handheld digital multimeter, autoranging, 6000 counts, in a dark grey rubber holster: 170 x 90
  x 44 mm, 450 g, lying face up on the table 0.17 m in front of the arm bases on their midline, its display end
  toward the strip (17 mm short of the blocks), held to the table by a hook-and-loop strip under the holster. Its display, a
  46 x 74 mm LCD at the far end, shows a minus sign and four 22 mm seven-segment digits with decimal points. Its
  rotary selector, 30 mm nearer the arms than the face's middle, is a 40 mm round knob with a grip bar across it
  (38 x 8 mm, 9 mm tall, a white line at its pointer end); it clicks between five positions 30 degrees apart,
  clockwise seen from above: **OFF** (where it starts), **V~**, **V DC** (pointer straight at the display), **mV**
  and **Ohm**, with a stop at each end; about 0.12 N m turns it from one position to the next. White ticks and labels
  on the face mark the positions. Two 4 mm input jacks at its near end: COM (black) and V (red).
- **Probe holders:** two black plastic blocks, 30 x 30 x 30 mm, fixed to the table 0.20 m in front of the arm bases,
  0.14 m left and right of their midline, each with an 8 mm square hole in its top.
- **Probes:** a red one in the left holder and a black one in the right, standing tip down. Each is 140 mm long,
  30 g: a 2 mm steel tip 15 mm long, a 6 mm shroud 10 mm long (in the holder's hole), a 16 mm finger guard resting
  on the holder's top, a 10 mm handle 100 mm long and a 6 mm cap. Each has a 1 m silicone lead with a 4 mm shrouded
  plug: the red one in the meter's V jack, the black one in COM.

### Real-world trial: equipment

- **Meter:** any handheld autoranging digital multimeter of at least 6000 counts with a rotary selector whose
  positions, clockwise from OFF, begin V~, V DC (some meters label it V with a solid and a dashed line), then others
  (mV, Ohm and so on); DC volts accuracy about ±(0.5 % + 2 digits); CAT III 600 V rated; in its rubber holster; with
  a display at least 40 mm wide. Selector, display and jacks as above (a typical 30 to 60 US dollar meter). Stick it
  down with a 50 mm strip of hook-and-loop tape at the position above.
- **Probe leads:** a pair of silicone test leads, 1 m, red and black, with 4 mm shrouded plugs and probes with a
  2 mm tip 15 mm long (the long tips sold with most meters; any tip caps removed).
- **Terminal strip:** 40 feed-through screw-clamp terminal blocks for 2.5 mm² wire with a test socket (2.3 to 2.6 mm)
  on top, on a 500 mm TS35 rail screwed to a board fixed to the table, colours as above.
- **Test-voltage source:** a regulated 12 V DC plug-in power adapter (switch-mode, SELV, class II, at least 0.5 A,
  output 12 V ±5 %) with bare-ended output leads (or a DC-jack pigtail): the + wire stripped 8 mm and clamped in the
  L block's far wire entry, the - wire in the N block's, so that no bare copper shows outside the clamps. The adapter
  is plugged into a mains socket away from the robot. 12 V DC is extra-low voltage: safe to touch, nothing on the
  strip is at mains voltage. No other block is wired.
- **Probe holders:** two 30 mm cubes with an 8 mm square hole 26 mm deep in the top (3D-printed, or drilled
  hardwood), fixed to the table with double-sided tape.
- **Camera on the meter:** besides the trial cameras, a phone on a small stand over the meter (or the top camera,
  if it resolves the digits) records the display through the trial.

## Scoring

The score rests on the meter's own output: the readings it shows on its display. The meter takes a reading twice a
second; with the selector on V DC it shows the voltage between its V and COM jacks, which is 12.00 V (or -12.00 V
with the probes the other way round) only while the red probe's tip touches one of the L and N clamps and the black
probe's tip the other, both at once. With either tip off its clamp, it shows 0.00. On the other positions it shows
no DC voltage reading (OFF: blank; V~: 0.00; mV: OL once the voltage is over its 600 mV range, else 0.0; Ohm: OL).

At the end of the episode:

| credit | condition | real-trial check |
| --- | --- | --- |
| 0 | a probe tip **entered any other terminal's test opening** at any point (1 mm or more below its mouth, inside its bore) | the trial video of the strip: a tip seen going into another block's opening |
| 1.0 | otherwise, **measured**: with the selector on V DC, the display showed the **test voltage** (11.40 to 12.60 V, either sign) at **three readings in a row** (the voltage on the display for at least 1 s), with both probes held in the grippers at each of those readings (success) | the meter's display in the video: 12 V shown without a break for 1 s; the probes in the grippers |
| 0.5 | otherwise, the display showed the test voltage at one reading at least (both tips on their clamps at once, briefly), with both probes held | the meter's display in the video: 12 V shown, however briefly |
| 0.25 | otherwise, a probe held in a gripper had its tip put into the L or N test opening, 4 mm or more below its mouth (half the opening's depth: at most 11 mm of its 15 mm tip still showing) | the trial video of the strip, side view |
| 0 | otherwise | |

- **Contact:** a tip touches a terminal's clamp when it is inside that opening's bore and its end is within 0.3 mm of
  the opening's floor (the clamp), 8 mm below the mouth. A tip only part way down the opening does not touch the
  clamp, and the meter shows 0.00.
- **Held:** a closed gripper on the probe's handle: the gripper's centre within 4 mm of the probe's axis, along the
  handle, its pads closed to the handle's width plus 1.5 mm at most. (A probe left standing in an opening on its own
  does not count: the probe is in the robot's hand.)
- The task file keeps a record, every control step (20 Hz), of each tip's position, which opening it is in and how
  deep, whether it touches the clamp, whether its probe is held, any entry into a wrong opening, the selector's
  position, and every reading the meter showed (twice a second, at 0.5 s, 1.0 s, ... of the run); the score is read
  from it.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,600 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 550) | selector at V DC by step 170 (two clicks, OFF to V DC); the red probe lifted out at step 202, the black one (after the selector, done at step 219) at step 360; the red tip into L and the black tip into N at the first try, each pressed down onto its clamp (8.0 mm below the mouth); first 12.00 reading at step 530, then 12.00 at three readings in a row (530, 540, 550: 1 s); no tip entered any other opening |
| GPU, `--variant brief` | **0.5** (634 steps) | 12.00 shown at two readings, then the black probe came out: 0.00 |
| GPU, `--variant nodial` | **0.25** (657 steps) | meter left at OFF: display blank with both tips on their clamps |
| GPU, `--variant wrong` | **0** (789 steps) | the black tip entered the opening of the beige block two from N (wrong opening): 0 from then on, although 12.00 was shown afterwards |
| GPU, `--variant hold` | **0** (1,600 steps) | no arm moves; display blank |
| CPU (`--device cpu`), all variants | 1.0 (success at step 520) / 0.5 / 0.25 / 0 / 0 | the same sequences |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored **0.0**: in 1,600
steps its arms moved about over the table and toward the probes, but it never grasped a probe or turned the selector;
the display stayed blank.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
position), never the simulator's object poses. The right arm first switches the meter on: it comes down over the
selector with its gripper open across the knob's grip bar, closes on the bar, turns its wrist 60 degrees clockwise
(two clicks: OFF, V~, V DC), lets go and rises. Each arm then, the gripper pointing down with its fingers closing
across the rail: comes down over its probe's handle, closes on it 80 mm above the tip (near its centre of mass, so it
does not pivot in the grip), lifts it 50 mm out of its holder, carries it over its terminal's test opening, lines its
measured gripper position up over the opening to within 0.4 mm, and lowers the tip at 6 mm/s, its command never more
than 3 mm ahead of the gripper, until the tip is 5 mm below the opening's mouth. It feels for the opening: if the
gripper stops descending early (the tip is on the block's top, beside the opening), it lifts the tip 4 mm, shifts 1
or 2 mm and tries again, up to nine times. Once the tip is in, it goes on down until the gripper stops (the tip is on
the clamp) and holds it there, pressing lightly. The left arm takes the red probe to L, the right arm the black probe
to N. The meter then shows 12.00 V.

The variants check the scoring: `brief` takes the black probe out again about 0.8 s after both tips are on their
clamps (one or two readings of 12.00: 0.5); `nodial` leaves the selector at OFF, so the display stays blank with both
tips on their clamps (0.25); `wrong` first puts the black probe into a beige terminal two blocks from N, then into N
(0 from the moment the tip enters the wrong opening).

The videos show the top camera with a close-up of the meter cut from the same top-camera frame beside it, and the two
wrist cameras below.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Strip and holders:** fixed colliders made of boxes, with small wedges for the openings' lead-ins.
- **Probes:** free rigid bodies; their collision shapes are square sections (tip, shroud, guard, handle, cap).
- **Meter:** an articulation with a fixed root (the case, a box collider) and the selector knob on a revolute joint
  with stops at OFF and Ohm; its grip bar is a box collider. The click is the joint's drive (0.46 N m/rad), its
  target re-aimed every control step at the nearest position: up to 0.12 N m half-way between two positions. The
  display's segments are thin dark quads: a shown one sits just over the LCD, a hidden one is moved under its
  surface, at every reading.
- **Record:** `probe_parts._record` runs once per control step, from the probes' poses, the selector's joint angle
  and each arm's joint readings (its gripper centre from the arm's forward kinematics, and its finger gap).

## Assets

No third-party assets: the rail, the terminal strip, the holders, the probes and the meter are this task's own
geometry, built in code (`probe_parts.py`, `meter_parts.py`, `tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- No electricity is simulated: a tip touches a clamp when it is in the bore within 0.3 mm of the opening's floor,
  read from the probe's pose; the meter's reading is the source's 12.00 V when both tips touch the L and N clamps,
  0.00 otherwise.
- The rendered display takes about 2 to 3 s to settle on a new reading (the renderer's temporal filter blends a
  changed digit in); the reading the score uses changes at once.
- The probes' leads are not simulated (about 1 m of light silicone cable each, from the probe's end to the meter).
- The meter's case is fixed in place (taped down in the real trial); its selector's click is a spring re-aimed at the
  nearest position.
- The terminal blocks are fixed colliders of a generic design; the test openings are square (2.6 mm), sized for the
  2 mm tips. The probe's collision shapes are square sections.
- Friction: probe handle rubber 0.9, meter holster rubber 0.9, other plastic 0.4, housings polyamide 0.35, robot
  fingers PhysX default 0.5.
