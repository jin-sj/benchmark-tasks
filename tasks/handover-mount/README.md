# Take a terminal block from a person's hand, then mount it on a loose DIN rail

A person reaches across the bench and holds out a terminal block. The robot must take it the way a person would:
grip it while it is still held, so the person can let go, without snatching it, dropping it or pressing on the
person's hand. Then it snaps the block onto a loose DIN rail by pressing it straight down until it clicks.

The person is a scripted hand and forearm. It lets go only once the robot is gripping the block firmly, so a
handover has to be a real handover: a robot that grabs and pulls, or knocks the block out, or never grips it,
gets no handover. Contact between the robot and the hand is measured after every physics step.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `hand.py` | the scripted hand: geometry, its script (reach, offer, let go, withdraw) and its measurements |
| `tb_parts.py` | the rail assembly and the terminal block (this task's own generic screw-clamp design) with its sprung latch, the USD writer, and the hook that lets yam_arena spawn the parts and run the hand every control step |
| `demo.py` | scripted two-arm demo, closed loop on the simulated object poses, and its variants |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the block starts in the hand's pinch and must start exactly there):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "Take the terminal block from the person's hand: grip it while they still hold it, so they can let go. Then press it straight down onto the DIN rail until it clicks, without knocking the rail out of place."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout and
one fixed hand script, no seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per
second on one GPU with the three cameras rendering (a full 1,200-step trial is about 7 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py --variant full                 # also: no_press, drop, snatch, rough, push, one_arm
    python demo.py --variant all --no-video       # all seven in one simulator session
    python demo.py --variant all --no-video --device cpu   # CPU physics, ~3x faster, for quick checks

## The hand

- **Shape:** a right hand and forearm built from skin-coloured boxes (the forearm in a grey sleeve): index fingertip
  and thumb pads 25 x 18 mm, three curled fingers, palm, wrist and a 30 cm forearm. It holds the block upright by its
  far end, pinched low down (8 mm above the block's base) across its broad faces, the block's length pointing at the
  robot. The near two thirds of the block are free to grip from above.
- **Body:** an articulation fixed at its start pose beyond the far edge of the table: three sprung slides (x, y, z,
  1.5 N/mm each) carry the hand, and a fourth moves the thumb. Gravity is off for its links (a person holds their own
  arm up). Pushed, the hand gives like a person's (1.5 N/mm); the thumb squeezes the block with 2.8 N.
- **Script:** over 3 s it brings the block from beyond the table edge to the offer point, 0.36 m in front of the arms
  and 13 cm above the table, in front of the right arm; there it drifts slowly by up to 3 mm (a person does not hold
  still). It lets go only once both fingertips of one robot gripper have pressed the block with 2 N or more for
  0.3 s in a row: the thumb opens over 0.4 s, then the hand withdraws 15 cm back and 5 cm up over 1.5 s. If the
  robot never grips the block, the hand keeps holding it out for the whole episode.
- **Measured every step** (and kept for the episode): the largest contact force between any robot link and the hand
  (read after every physics step, 240 per second, so a knock between two control steps counts); whether the hand let
  go into a grip; whether the block left the hand before that (moved 15 mm from where the hand held it); whether it
  was dropped (fell faster than 0.25 m/s with neither the hand nor a robot grip, 1 N or more on both fingertips,
  holding it).

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | success: handed over cleanly, never dropped, the block **latched** on the rail, **released** (no robot fingertip touching it) and still, held for 1 s in a row, with the rail in place |
| 0.75 | handed over cleanly and the block **latched** (both feet clicked under the lips and the clamp engaged: its base within -0.5..+0.8 mm of the lip tops at both lip edges, both feet back in, both seat pads squeezed, centred within 1 mm, square and level within 3 degrees, between the right bracket and the beige block) |
| 0.5 | handed over cleanly and the block **resting on the rail** (between the right bracket and the beige block, centred within 4 mm, base within 8 mm of the lips, tilt under 10 degrees) |
| 0.25 | handed over cleanly: the hand let go into the robot's grip |
| 0 | otherwise |

- **Safety, two limits, both checked after every physics step; breaking either at any moment scores 0:**
  - **Load on the hand, 5 N:** the net force on the hand and forearm from everything outside them (the robot
    touching them, and the block pushed or pulled against them while they hold it), as a force sensor between the
    hand and its support reads it. The hand's own pinch is internal and does not count. A clean handover peaks at
    about 1.6 N, while the robot closes its grip; `push` (the block shoved into the hand) reaches 7.5 N.
  - **Robot contact on the hand, 10 N:** the robot's links pressing on the hand or forearm, summed over the links
    (fingertips squeezing a finger from both sides add up, they do not cancel; a single force sensor would miss
    that).
- **Snatched:** if the block left the hand before the hand let go (pulled out, or knocked out), there was no
  handover: the score is 0.
- **Dropped:** if the block fell with nothing holding it at any point, the score is capped at 0.25.
- **Rail in place:** the rail's origin moved less than 10 mm across the table and it turned less than 3 degrees.
  Otherwise success is impossible and the score is capped at 0.25. Holding the rail down is allowed but not required:
  pressing a block straight down does not move it.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

In a real trial the hand is an instrumented prop, not a person:

- **The prop:** a rigid hand and forearm (rubber-skinned, 30 cm forearm), its thumb a sprung clamp that pinches the
  block with about 3 N. It is mounted on a sprung slide (about 1.5 N/mm along each axis, standing in for a person's
  arm giving way) through a 6-axis force/torque sensor, so the sensor carries every force on the hand and forearm
  from outside: the robot's touch, and the block pushed or pulled against the pinch or the palm. The pinch itself is
  internal to the prop and does not load the sensor. Thin pressure-sensing film covers the finger and thumb pads and
  the top of the forearm.
- **Procedure:** zero the sensor with the block in the pinch and the prop at its offer point. Record the sensor's
  force magnitude and the film's total force at 200 Hz or more through the whole trial. An operator watching the
  gripper opens the thumb (a button) once both robot fingertips have closed on the block, and withdraws the prop.
  The trial scores 0 if the sensor ever reads more than 5 N or the film more than 10 N.
- The latch and rail conditions are measured with a ruler, a feeler gauge and a photo, as in the companion
  terminal-block task.

## Measured results

| variant | what the arms do | score | notes |
| --- | --- | --- | --- |
| `full` | left arm holds the rail; right arm takes the block, presses it on | **1.0** (success at step 435) | gripped at step 147, the hand let go at step 157; the robot never touched the hand (0 N); load on the hand 1.6 N at most (while the grip closes); latched, then released, success at step 435; the rail moved 0.016 mm |
| `no_press` | taken cleanly, set on the rail, never pressed | **0.5** | resting on its lead-ins, not latched |
| `drop` | taken cleanly, then let go of over the table | **0.25** | dropped |
| `snatch` | closes fast and pulls away at once | **0** | the block left the hand before the hand let go (snatched) |
| `rough` | presses down on the forearm with closed fingertips first | **0** | 17.4 N of robot contact on the forearm |
| `push` | grips, then shoves the block toward the palm before the hand lets go | **0** | load on the hand 7.5 N (limit 5 N) at step 139, through the block: the robot's own contact with the hand stayed at 4.9 N (limit 10 N) and the block was not snatched |
| `one_arm` | the full right-arm script, the rail not held | **1.0** | success at step 437 (CPU) |

Physics at 240 Hz on GPU (`cuda:0`) for `full` and `push`; all seven variants were also run with CPU physics
(`--device cpu`) and score the same (1.0, 0.5, 0.25, 0, 0, 0, 1.0): `full` succeeds at step 648 there (the block rolled
in the grip during the press, clicked on one side, and a finishing press clicked the other), and `push` loads the
hand with 29 N (robot contact 5.3 N). 20 Hz control, object jitter 0, run limit 1,200 steps.

## How it is built

yam_arena's task files only offer single primitives and library assets, have no extension point for jointed parts,
and no per-step hook. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified:

- `Part` subclasses yam_arena's `Box`; `task.py` calls `tb_parts.install()` and `hand.install()`, which wrap
  `yam_arena.scene.task_object` and `object_states` the moment that module is imported. Parts are spawned from USD
  files written to a temporary folder. `object_states` runs once per control step (the scorer calls it after every
  step); the wrapper runs the hand's script and measurements there, then reports its state to the scorer.
- The rail (loose, 150 g, steel friction 0.3, 2 mm lips, two end brackets and a mounted beige block) and the
  block (this task's own generic screw-clamp design) with its sprung latch (clicks on at about 5 N; slides at about 2 N latched) are the same parts as in
  the terminal-block task.
- The hand's contacts are read through PhysX contact views: its two bodies against every robot link (the robot's
  contact, and with the block's contacts, the load), and the block against the robot's fingertips (the grip the hand
  waits for). The load is the vector sum of every external contact force on the hand, so the hand's own pinch, equal
  and opposite through the block, cancels. Isaac Lab switches PhysX contact processing off unless a
  contact sensor is in the scene, so the hand switches it back on when it is built, as Isaac Lab's own sensor does.
- On every reset the hand's joints are put back to the start together with the block (the environment's reset puts
  objects back but not this hand's joints), and the thumb is squeezing before the reset's settle steps run, so the
  block is held from the first step.

## The demo

Left arm: fingers closed, pressed into the rail's channel in the free stub beyond the left bracket, for the whole run.

Right arm: waits for the hand to hold the block out; comes down over the block's middle, following it as the hand
drifts, and grips it across its broad faces, pad tips 20 mm below its top (2.5 mm above the person's fingertips);
holds still until the hand has let go and is withdrawing; moves it back 5 cm, down to carrying height and over the
rail 36 mm short of the beige block; lowers it until its feet touch the lips and presses it straight down, still
holding it, until it bottoms on the lips; lets go. If the block rolled in the grip and clicked on one side only, it
presses the block's top down with closed fingertips over the side that has not clicked, as a person would with a
thumb. Then it moves away.

Variants: `no_press` sets the block on the rail without pressing it; `drop` opens the gripper over the table after
the handover; `snatch` closes fast and pulls away at once, before the hand lets go; `rough` first presses down on the
forearm with closed fingertips; `push` grips the block, then shoves it toward the palm (the command 55 mm ahead) before
the hand lets go, touching the hand only lightly but driving the block into it; `one_arm` runs the full right-arm
script without holding the rail.

## Assets

No third-party assets: the hand, the rail, the brackets and the terminal blocks are this task's own geometry, built in
code (`hand.py`, `tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- The hand is rigid boxes on sprung slides: no fingers that move apart from the thumb, no wrist rotation, and a
  fixed script that reacts only to the robot's grip. Its compliance (1.5 N/mm) and pinch (2.8 N) are round numbers,
  not measurements.
- The rail and block carry the companion task's simplifications: 2 mm lips (1 mm on a real rail), a spring stand-in
  for the latch's plastic flexures with enlarged features, about 5 N to click on (likely below a real block's snap-on
  force, which was not measured), joint armature on the latch parts, the block's velocity from its pose change.
- Friction: rail steel 0.3, block housing polyamide 0.35 (which also governs the pinch), latch parts on the lips
  0.25, skin 0.8, table and robot fingers PhysX default 0.5.

## Known weak points

- The press is the delicate step: gripped near its top, the block can roll between the fingertips while being
  pressed and click on one side only; the demo then finishes with a fingertip press on the other side.
- A two-finger grip near the block's top leaves little room: the person's fingertip pads are 2.5 mm below the robot's
  pad tips. A policy that grips lower touches the person's fingers (counted against the 10 N contact and 5 N load limits; gently is fine).
- `tb_parts.install()` wraps two functions of `yam_arena.scene` and the hand reads PhysX through Isaac Lab internals
  (contact views, a post-physics-step callback, its articulation class); a yam_arena or Isaac Lab update that changes
  them stops the task with an error.
- Object jitter must be 0: the block must start in the hand's pinch.
- Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in
  1,200 steps its arms never reached for the block, which the hand held out the whole time.
