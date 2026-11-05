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
| `tb_parts.py` | the rail assembly and the WDU-style block with its sprung latch, the USD writer, and the hook that lets yam_arena spawn the parts and run the hand every control step |
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

    python demo.py --variant full                 # also: no_press, drop, snatch, rough, one_arm
    python demo.py --variant all --no-video       # all six in one simulator session
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
| 1.0 | success: handed over cleanly, never dropped, the block **latched** on the rail and still, held for 1 s in a row, with the rail in place |
| 0.75 | handed over cleanly and the block **latched** (both feet clicked under the lips and the clamp engaged: its base within -0.5..+0.8 mm of the lip tops at both lip edges, both feet back in, both seat pads squeezed, centred within 1 mm, square and level within 3 degrees, between the right bracket and the beige block) |
| 0.5 | handed over cleanly and the block **resting on the rail** (between the right bracket and the beige block, centred within 4 mm, base within 8 mm of the lips, tilt under 10 degrees) |
| 0.25 | handed over cleanly: the hand let go into the robot's grip |
| 0 | otherwise |

- **Safety:** if the robot ever pressed on the hand or forearm with more than 10 N, the score is 0.
- **Snatched:** if the block left the hand before the hand let go (pulled out, or knocked out), there was no
  handover: the score is 0.
- **Dropped:** if the block fell with nothing holding it at any point, the score is capped at 0.25.
- **Rail in place:** the rail's origin moved less than 10 mm across the table and it turned less than 3 degrees.
  Otherwise success is impossible and the score is capped at 0.25. Holding the rail down is allowed but not required:
  pressing a block straight down does not move it.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

In a real trial the hand is a prop hand and forearm on a force gauge (or a person's hand in a pressure-sensing
glove), and the person lets go when they feel a firm grip; the latch and rail conditions are measured with a ruler, a
feeler gauge and a photo, as in the companion terminal-block task.

## Measured results

Physics at 240 Hz on GPU (`cuda:0`), 20 Hz control, object jitter 0, run limit 1,200 steps.

| variant | what the arms do | score | notes |
| --- | --- | --- | --- |
| `full` | left arm holds the rail; right arm takes the block, presses it on | **1.0** (success) | robot-hand contact 0 N throughout; gripped at step 147 (11.5 N per fingertip), the hand let go at step 157; latched, success at step 394; tilt 0.2 degrees, rail moved under 0.01 mm |
| `no_press` | taken cleanly, set on the rail, never pressed | **0.5** | resting on its lead-ins 4.7-4.9 mm above the lips |
| `drop` | taken cleanly, then let go of over the table | **0.25** | dropped at step 226 |
| `snatch` | closes fast and pulls away at once | **0** | the block left the hand at step 137, 3 steps after the grip, before the hand let go |
| `rough` | presses down on the forearm with closed fingertips first | **0** | 17.4 N on the forearm at step 104 |
| `one_arm` | the full right-arm script, the rail not held | **1.0** | success at step 394; the rail moved 0.009 mm at most |

With CPU physics (`--device cpu`) the six variants score the same (1.0, 0.5, 0.25, 0, 0, 1.0); there the block rolled
in the grip during the press, clicked on one side, and the finishing press clicked the other (success at step 615).

Success registers while the gripper is still pressing the latched block (the scorer cannot tell whether a hand is
touching it); in the demo the gripper lets go right after, and the block stays latched.

## How it is built

yam_arena's task files only offer single primitives and library assets, have no extension point for jointed parts,
and no per-step hook. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified:

- `Part` subclasses yam_arena's `Box`; `task.py` calls `tb_parts.install()` and `hand.install()`, which wrap
  `yam_arena.scene.task_object` and `object_states` the moment that module is imported. Parts are spawned from USD
  files written to a temporary folder. `object_states` runs once per control step (the scorer calls it after every
  step); the wrapper runs the hand's script and measurements there, then reports its state to the scorer.
- The rail (loose, 150 g, steel friction 0.3, 2 mm lips, two end brackets and a mounted beige block) and the WDU
  2.5-style block with its sprung latch (clicks on at about 5 N; slides at about 2 N latched) are the same parts as in
  the terminal-block task.
- The hand's contacts are read through PhysX contact views. Isaac Lab switches PhysX contact processing off unless a
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
forearm with closed fingertips; `one_arm` runs the full right-arm script without holding the rail.

## Fidelity: what is simplified

- The hand is rigid boxes on sprung slides: no fingers that move apart from the thumb, no wrist rotation, and a
  fixed script that reacts only to the robot's grip. Its compliance (1.5 N/mm) and pinch (2.8 N) are round numbers,
  not measurements.
- The 10 N limit applies to contact with the hand and forearm; force passed through the block while both hold it
  (a tug) is not counted, but pulling the block out before the hand lets go is (snatched).
- The rail and block carry the companion task's simplifications: 2 mm lips (1 mm on a real rail), a spring stand-in
  for the latch's plastic flexures with enlarged features, about 5 N to click on (likely below a real block's snap-on
  force, which was not measured), joint armature on the latch parts, the block's velocity from its pose change.
- Friction: rail steel 0.3, block housing polyamide 0.35 (which also governs the pinch), latch parts on the lips
  0.25, skin 0.8, table and robot fingers PhysX default 0.5.

## Known weak points

- The press is the delicate step: gripped near its top, the block can roll between the fingertips while being
  pressed and click on one side only; the demo then finishes with a fingertip press on the other side.
- A two-finger grip near the block's top leaves little room: the person's fingertip pads are 2.5 mm below the robot's
  pad tips. A policy that grips lower touches the person's fingers (counted against the 10 N limit, gently is fine).
- `tb_parts.install()` wraps two functions of `yam_arena.scene` and the hand reads PhysX through Isaac Lab internals
  (contact views, a post-physics-step callback, its articulation class); a yam_arena or Isaac Lab update that changes
  them stops the task with an error.
- Object jitter must be 0: the block must start in the hand's pinch.
- Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in
  1,200 steps its arms never reached for the block, which the hand held out the whole time.
- The scorer cannot see whether the robot is still touching the block when it counts success.
