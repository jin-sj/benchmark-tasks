# Mate a sealed harness connector on the assembly line so its CPA lock will close

On vehicle and electrical-equipment assembly lines, assemblers mate sealed harness connectors onto sensors and modules
already installed on the sub-assembly, many times a shift, by hand and with no tool. A plug pushed on short of its
latch makes contact at first and works loose later, so modern sealed connectors carry a CPA (connector position
assurance) lock: a small slider that only closes once the plug is fully on and its primary latch has caught. The
robot must connect the harness plug to the sensor and close its CPA.

Who does this (BLS Occupational Employment and Wage Statistics, May 2024, as quoted in BLS The Economics Daily,
29 Sep 2025):

| occupation | O*NET | where | US jobs |
| --- | --- | --- | --- |
| miscellaneous assemblers and fabricators, incl. team assemblers | 51-2092.00 | motor vehicle manufacturing | 152,670 |
| the same | 51-2092.00 | motor vehicle parts manufacturing | 138,530 |
| electrical, electronic and electromechanical assemblers | 51-2028 (51-2022.00, 51-2023.00) | manufacturing | 228,530 |

The O*NET task lines: 51-2023.00 "Connect cables, tubes, and wiring, according to specifications"; 51-2022.00
"Assemble electrical or electronic systems or support structures and install components, units, subassemblies,
wiring, or assembly casings"; 51-2092.00 "Provide assistance in the production of wiring assemblies" and rotating
through the line's stations. A vehicle carries a few hundred harness connectors, most of them mated by hand on the
line and in module and harness plants, so connector mating is a routine station operation for these workers, about
520,000 in all.

## What the task tests

Pressing a part home until it clicks is an ability many tasks already cover. What this task tests is the
**two-stage lock**: the CPA only goes home once the primary latch is fully seated, so the outcome shows whether the
mating was complete, and a half-mated plug cannot be locked. To get there the robot has to:

- **Seat the plug fully with a firm, straight axial push.** The plug only latches when pushed straight and fully home:
  a steady push along the axis latches it at 32.9 N (measured, `force_test.py`; the terminals' and seal's friction,
  the seal's compression and the latch arm together, as on a real sealed 3-way connector), delivered by the arm.
  Pushed with 25 N it stops on the lock ramp, half-mated, and stays there when let go; pushed crooked, it cocks in
  the 0.25 mm-clearance shroud and binds short of the latch.
- **Then close the CPA, which only moves on a latched plug.** On a half-mated plug the CPA stays in pre-lock however
  it is pressed (its pre-lock is released by the header only when the plug is home, and its lower end lands on the
  spread latch arm); on a latched plug it goes home at 15 N.
- **Handle a part on a flexible lead.** The plug is the end of a 190 mm harness lead clipped to the harness trunk: it
  has to be picked up, turned onto the header's axis and pushed on against the lead's pull, and the lead tilts the
  plug as soon as it is let go in the header's mouth.

The header is where it is on the line: a sensor module on its bracket on the sub-assembly, between a relay box and
a coolant hose already fitted, so the plug can only come in from above and from the robot's side.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), the CPA's pre-lock and detent (once per control step), `success`, `score`, `distance`; docstring = what is on the table |
| `cpa_parts.py` | the sensor module with its header, the plug with its latch arm and CPA, its lead and clip; their geometry, joints and springs |
| `tb_parts.py` | shared shape and USD helpers, and the hook that lets yam_arena spawn the parts and run the task's step hook |
| `demo.py` | scripted demo, using only the known layout and the robot's joint readings |
| `force_test.py` | mechanism test: a known axial force on the plug, to measure the force at which it latches and check a half-mated plug stays put |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1800 \
        --instruction "Connect the harness plug to the sensor and close its orange CPA lock."

`task_score` gives 1.0 when the episode ends in success (held 2 s), otherwise `score()` at the last step. One fixed
layout, no seeds needed. Runtime: about 30 s of simulator start-up, then about 2 control steps per second on one GPU
with the three cameras rendering (a full 1,800-step trial is about 15 minutes).

Scripted demo:

    python demo.py                      # GPU physics, as a trial runs: 1.0
    python demo.py --variant nocpa      # plug on and latched, CPA left in pre-lock: 0.5
    python demo.py --variant half       # plug pushed only onto the lock ramp and left there: 0
    python demo.py --variant halfcpa    # plug left half-mated, then the CPA pressed with up to 25 N: 0
    python demo.py --variant hold       # no arm moves: 0
    python demo.py --variant all        # all of them, one simulator session
    python demo.py --device cpu         # CPU physics, faster, for quick checks

Mechanism test (not a demo: a known force applied straight down the plug's axis, after the demo's arm has let go of
it in the shroud's mouth):

    python force_test.py ramp           # force rising 2 N/s from 0: the force at which the plug latches
    python force_test.py hold25         # 25 N for 4 s, then released: stays unlatched and does not back out

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left.

- **Sensor module**: cast aluminium, 44 x 52 mm, 62 mm tall, on a steel bracket plate 70 x 90 x 4 mm bolted to the
  sub-assembly's line pallet (the table top), the plate centred at x 0.36 m, y -0.12 m.
- **Header**: on the module's top, a black PBT shroud, cavity 14.5 x 22.5 mm, 17 mm deep (rim 83 mm above the
  table), 1.8 mm walls with a 1 mm lead-in chamfer at the mouth, a 4 mm keyway in its left wall; three pins and a
  blue interface seal inside. On its wall facing the robot, the lock ramp: 2.0 mm proud, a 30 degree lead-in face
  above a flat catch face 9.0 mm above the cavity floor.
- **Neighbours**: a grey plastic relay box (50 x 38 mm, 105 mm tall) 47 to 85 mm to the left of the header's axis;
  a black rubber coolant hose (28 mm across, 140 mm long, running left-right) 48 mm behind the axis, 48 to 76 mm
  above the table.
- **Harness plug**: black PA66, about 14 g: a 14 x 22 mm body that enters the shroud, a collar 18.4 x 24.8 mm from
  16 to 30 mm above its face (it stops on the shroud's rim), a key rib on its left side, a blue radial seal on its
  far side (8 to 15 mm above its face). On its side facing the
  robot, the latch arm (two rails and a hook crossbar, sprung toward the housing at 15 N/mm) and, outside it, the
  orange CPA slider (top 3 x 10 mm), standing 4 mm proud of the plug's top in its pre-lock position; on its far
  side a 90 degree wire-dress cover. It stands on its face on the table at x 0.26 m, y -0.20 m, turned like the
  header.
- **Lead and clip**: 190 mm of 7 mm black convoluted tube (three wires) from the dress cover, straight away from
  the robot 24 mm above the table, to a white nylon harness clip (18 mm tall) on a 24 x 24 x 15 mm steel block at
  x 0.463 m; the clip also holds the harness trunk (12 mm, taped) running left-right.

## The connector's mechanism

- **Mating**: from 9 mm above the cavity floor the plug's radial seal (a pad on its side, sprung outward at 12 N/mm
  with a 1.05 mm preload, pressing about 14 N on the cavity wall) grips the cavity, friction 0.35 on both walls: it
  holds the plug against moving either way, as the terminals' and seal's drag does on the real part. The plug's face meets the seal plate 6.0 mm
  above the floor and goes on 5.0 mm more to its seated position (the collar on the rim, the face 1.0 mm above the
  floor), compressing it (1.5 N/mm, 7.5 N seated).
- **Primary latch**: from 3.5 mm above the seated position, the hook rides up the lock ramp's lead-in face, spreading
  the latch arm 1.35 mm (15 N/mm); within 0.6 mm of seated it clears the catch face and snaps in under it. Latched,
  the plug cannot be pulled off. Not latched, it stays where it was left (the seal's grip): half-mated, its latch arm
  still spread.
- **Measured** (`force_test.py`, CPU physics, a force straight down the plug's axis): the plug does not move until
  about 21 N; rising 2 N/s, it latches at 32.9 N. Held at 25 N for 4 s, it stops 1.34 mm above its seated position
  with the latch arm spread 0.71 mm, not latched; released, it stays there (1.36 mm, 18 s later).
- **CPA**: rests in pre-lock or home by a detent (15 N to move it from either). It is held in pre-lock until the
  plug is latched (the real part's pre-lock finger is released by a rib on the header only when the plug is home),
  and with the latch arm spread its lower end lands on the arm's top. Home (4.0 mm down, flush with the plug's top),
  it lies outside the latch arm and keeps it from spreading.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | **latched** and **CPA home** (success, held 2 s) |
| 0.5 | latched, the CPA not home |
| 0 | not latched: half-mated, off the header, or never connected |

- **Latched**: the plug on the header's axis (its face's middle within 1 mm of it), its collar within 1.5 mm of the
  shroud's rim, its latch arm sprung back to within 0.5 mm of rest (the hook under the lock ramp's catch face).
- **CPA home**: pushed down at least 3.0 mm of its 4.0 mm travel.

In a real trial: a side photo shows the collar down on the rim and the latch arm flat (a half-mated plug's arm stands
spread on the ramp), and the plug stays on when pulled by its body by hand (a half-mated one pulls off); a photo shows the CPA flush with the plug's top (within 1 mm)
or still standing 4 mm proud.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,800 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 554) | plug gripped by step 123, let go in the shroud's mouth by 307; pushed on, latched at step 402; CPA home by step 515; still latched with the CPA home at the end, collar 0.64 mm above the rim |
| CPU | **1.0** (success at step 589) | the same sequence |
| CPU, `--variant nocpa` | **0.5** | latched (collar 0.25 mm above the rim, latch arm back to 0.0001 mm), CPA left in pre-lock |
| CPU, `--variant half` | 0 | pushed only onto the lock ramp and let go: half-mated, the collar 2.03 mm above the rim, the latch arm spread 0.41 mm, not caught; it stays there |
| CPU, `--variant halfcpa` | 0 | left half-mated as in `half`, then the CPA pressed with up to 25 N (the arm's estimate; more than the 15 N that moves it on a latched plug): the plug goes on to 1.04 mm above the rim, its latch arm spread 0.87 mm, not caught; the CPA stays in pre-lock (at most 0.003 mm of travel) |
| CPU, `--variant hold` | 0 | no arm moves |

The arm's own estimate of its push (from its joint drives' stiffness and position errors, leaving out its dynamics)
reads 22 N at the moment the plug latched on GPU (24 N on CPU), under the 32.9 N the force test measures for a steady
push. Either is within what the YAM's arm gives (its shoulder joints are rated 27 N m; 40 N at the fingertips 0.4 m
out needs 16 N m); the push goes through the closed fingertips, not the grip.

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 1,800
steps its right arm moved toward the plug on the table and hovered near it, but never took hold of it; the plug stayed
where it stood.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured TCP and an
estimate of the force it pushes with, from its joint drives' stiffness and position errors), never the simulator's
object poses.

The right arm, pointing down, its fingers closing across the plug's collar (closed in first to just clear of it,
then squeezed with the finger drives short of their cap so the two fingers centre the plug), picks the plug up,
lifts it 100 mm, carries it over the header (2.4 mm toward the robot's far side, for the tilt the lead gives the held
plug) and lowers it into the shroud until it rests on the seal, then opens wide and lets go. It closes its
fingers and pushes the plug's top straight down with its fingertips, 3 mm toward the latch side of its middle: the
command runs ahead of the arm's measured position (the arm gives under load, as the real one does) until the
measured fingertips are down at the seated height, then holds 0.6 s and eases off. Last it moves over the CPA and
pushes it home with its fingertips. The left arm only moves to the front left of the header, its wrist camera
looking at the plug's latch side from about 15 cm away, so that the latch and the CPA are in view.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Sensor and header**: one articulation fixed to the table: the bracket, module, shroud, ramp and neighbours, and
  the seal plate on a sprung prismatic joint.
- **Plug**: one articulation rooted at the clip (fixed): the four links of the lead on two-axis bend joints (0.03
  N m/rad each), a twist joint at the plug, the plug's housing, its latch arm (prismatic, sprung), its CPA
  (prismatic) and its radial seal pad (prismatic, sprung outward with a preload). The task's step hook opens the CPA's joint limit when the plug is latched and sets the detent's rest.

## Real-world setup

The parts are a common off-the-shelf type: a 3-way sealed automotive connector pair with a CPA lock and 1.5 mm
blade terminals, sold as pigtail kits for about US$10 to 20 a pair. The header half is held in a 3D-printed module
block of the size above, on a steel plate screwed to a board that stands for the line pallet; the relay box and hose
are any box and 28 mm hose of those sizes. The scene's geometry is this task's own generic design sized like such a
connector; its forces follow published figures for that connector class (terminal mating force about 12 N per
contact; maximum mating force 45 N (USCAR-25 class 2) to 75 N in makers' datasheets; CPA force with the connector
mated at most 22 N, and a CPA must not close on an unmated connector even at 80 N, USCAR-2).

## Assets

No third-party assets: every part is this task's own geometry, built in code (`cpa_parts.py`); no meshes, scans or
CAD files are used.

## Fidelity

- The latch arm's bending is a prismatic joint sprung at 15 N/mm. The terminals' and seal's drag is the contact
  friction (0.35) of a pad on the plug's side, sprung outward at 12 N/mm with a 1.05 mm preload (about 14 N on
  the cavity wall); the seal's compression is a plate on a 1.5 N/mm spring, which follows the plug back up when it is drawn
  out.
- The CPA's detent is a spring with two rests (15 N to move it); its release by the header is its joint limit,
  opened when the plug is latched.
- The lead is four rigid links on sprung bend joints.
- Friction: housing PA66 0.35, latch hook and ramp 0.25, hose rubber 0.9, metal 0.3; robot fingers PhysX default 0.5.
