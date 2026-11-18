# Replace a worn roller in a gravity roller conveyor

Roller conveyors carry the cartons and totes of warehouses, distribution centres and factory lines. Their rollers
wear (flat spots, scored tubes, seized bearings) and are swapped in place, as routine upkeep, by the plant's or
warehouse's maintenance staff. On a light-duty gravity conveyor each roller sits on a spring-retained hex axle whose
ends stick through hex holes in the two side rails: to take a roller out, the axle is drawn in at one end against its
spring until that end clears its rail, then the roller is swung up and lifted out; to fit one, one axle end goes into
its hole, the axle is drawn in, the roller is swung down between the rails and the axle let go so that it springs out
into the other hole, which it only does if its hex is turned to the hole's.

The robot must replace the worn roller of an installed conveyor section, between its neighbouring rollers, and put the
worn roller in the reject tray.

**Who does this, and how often.** General Maintenance and Repair Workers (O*NET `49-9071.00`, ~1,500,000 US workers;
listed tasks *"Repair machines, equipment, or structures, using tools ..."* and *"Dismantle machines, equipment, or
devices to access and remove defective parts ..."*), Industrial Machinery Mechanics (`49-9041.00`, ~415,000 US workers;
*"Repair or replace broken or malfunctioning components of machinery or equipment"*, *"Disassemble machinery or
equipment to remove parts and make repairs"*), Machinery Maintenance Workers (`49-9043.00`, ~80,000) and Millwrights
(`49-9044.00`, ~40,000), as well as warehouse and dock Hand Laborers and Material Movers (`53-7062.00`, ~3,000,000 US
workers). Counted only in the core sectors where gravity roller conveyors move cartons and totes daily — manufacturing
(474,330 maintenance mechanics), warehousing and storage (32,750), wholesale trade (106,960) and couriers (3,560) — the
four maintenance occupations alone number about 618,000 (BLS OEWS, May 2023). A distribution centre, loading dock or
factory packing line runs hundreds to thousands of unpowered gravity rollers, so popping out a flat-spotted or seized
spring-axle roller and snapping a replacement in by hand (with no tools) is routine work.

**What makes it hard, beyond holding a part or a round spring spindle.** Unlike a round spindle in open cups, a
conveyor roller has a **non-round 7/16 in hex axle (11.1 mm across flats) inside a freely spinning outer steel tube**
that must enter **closed 11.9 mm across-flats hex holes** in the frame webs (it can only enter within about $\pm 8^\circ$
of flat alignment). Because the outer tube spins freely on its ball bearings, turning the tube does not turn the axle:
the axle's hex flats must first be registered, then one hand must feed the left hex end blindly inside the 63.5 mm rail
channel under the roller into the left hex hole (between neighboring rollers only 66 mm away), and the other hand must
pinch the 25.4 mm protruding hex stub and draw it out 16 mm against its 13.6 N internal spring so the roller can swing
down between the rails before releasing the spring to drive the far hex end into the opposite hex hole.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `roller_parts.py` | the conveyor section (frame and its other rollers), the rollers (articulations: tube, bearings, sprung hex axle), the trays |
| `tb_parts.py` | shared shape and USD helpers, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted two-arm demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 4500 \
        --instruction "Replace the worn roller with the new one, and put the worn roller in the reject tray."

`task_score` gives 1.0 when the episode ends in success (held 1 s), otherwise `score()` at the last step. One fixed
layout, no seeds. Runtime: about 30 s of simulator start-up, then about 2 control steps per second on one GPU with the
three cameras rendering.

Scripted demo:

    python demo.py                      # GPU physics, as a trial runs: 1.0
    python demo.py --variant one_end    # new roller's left end in its hole, right end left on the rail: 0.5
    python demo.py --variant removed    # worn roller in the reject tray, new one not fitted: 0.25
    python demo.py --variant hold       # no arm moves: 0
    python demo.py --variant all        # all of them, one simulator session
    python demo.py --device cpu         # CPU physics, for quick checks

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left, z up from the
table top. The conveyor runs away from the robot; its top of rollers is 68.6 mm above the table (0.82 m above the
floor, a usual conveyor height).

- **Conveyor section** (installed, bolted down on the table): a standard light-duty gravity roller conveyor, 15 in
  (381 mm) between frames, 1.9 in rollers on 4-1/2 in (114.3 mm) centres. The frame is two formed steel angle rails,
  12 gauge (2.7 mm), painted blue: a web 63.5 mm tall standing up, a 38.1 mm leg out along the table, 610 mm long,
  tied at each end by a steel angle. The rails' webs face each other 381 mm apart (inner faces at y = +-190.5 mm);
  the section runs from x = 0.129 to 0.739 m. Each web has five 7/16 in hex holes (11.9 mm across flats, flats on top
  and bottom), 44.5 mm above the table, at x = 0.186, 0.300, 0.414, 0.529 and 0.643 m.
- **Rollers** (five, one in each pair of holes): galvanized steel tube, 1.9 in (48.3 mm) across, 16 gauge, 368.3 mm
  long, on two ball bearings, on a 7/16 in hex steel axle (11.1 mm across flats, 419.1 mm long, ends chamfered
  0.5 mm) that sticks out 25.4 mm past each end of the tube and 16.4 mm past each rail's outer face. 1.16 kg each. The
  axle is spring-retained: it slides through the bearings up to 16 mm toward the roller's left end, against a spring
  (4 N to start, 13.6 N at full travel), and springs back; it cannot slide the other way. The tube spins freely on
  the axle.
- **Worn roller**: the second from the robot (x = 0.300 m): dull and rust-stained, a flat scored band round its
  middle, a red paint mark near its left end. Its neighbours' tubes are 66 mm away on either side.
- **New roller**: the same roller, bright galvanized, lying in the parts tray along x, its left end (the end its axle
  slides toward) away from the robot, its axle's hex turned 15 degrees from flats-level.
- **Parts tray** (blue plastic) and **reject tray** (grey plastic): each 466 x 86 mm outside, 460 x 80 mm inside,
  25 mm walls (3 mm), 3 mm floor, long side along x, centred at (0.33, -0.30) and (0.33, -0.42) m: to the right of the
  conveyor, the parts tray nearer it. The reject tray is empty.

Real hardware: any commercial light-duty gravity conveyor section of this type (15 in between frames, 1.9 in 16 ga
rollers with spring-retained 7/16 in hex axles; sold in 5 ft and 10 ft lengths for a few hundred US dollars; spare
rollers about USD 20 each), bolted to the bench with its near end 0.13 m in front of the arm bases; its far part is out
of the arms' reach and plays no part. Two shallow plastic parts trays. The worn roller can be any spare roller marked
with paint.

## Scoring

At the end of the episode (success, held for 1 s, ends it early):

| credit | condition |
| --- | --- |
| 1.0 | the worn roller lies in the reject tray **and** both of the new roller's axle ends are seated in the slot's holes |
| 0.5 | the worn roller lies in the reject tray and one axle end of the new roller is seated |
| 0.25 | the worn roller lies in the reject tray |
| 0 | otherwise (the worn roller anywhere else, including still in the conveyor) |

- **Lies in the reject tray**: the worn roller's tube has its middle inside the tray's walls and rests on the tray's
  floor (its middle less than 5 mm above where it is when lying on the floor: not held above it).
- **An axle end is seated**: the axle passes through the slot's hex hole in that rail (where the axle crosses the
  rail's web, within 3 mm of the hole's centre: in the hole) and its end sticks out at least 5 mm past the rail's
  outer face. Either end may be in either rail. A roller lying on top of the rails or of other rollers, or with an end
  resting on a rail's top, has no end seated.
- The slot is the worn roller's: the other four positions are occupied by their own rollers.

Real-trial checks (a person, after the trial, with a ruler and a photo): the reject tray photographed from above shows
the worn roller lying in it; at each rail, the new roller's axle end is seen through the slot's hex hole from outside
the rail and a ruler laid on the rail's outer face shows it sticking out 5 mm or more (seated, it sticks out 16 mm);
lifting the roller's middle gently by hand does not lift either end out (a seated end cannot leave its hole).

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 4,500 steps (225 seconds).

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 2,410) | worn roller out and in the reject tray by step 711; new roller's left hex end seated in its hole at step 1,411; right hex end sprang into its hole at step 2,390 |
| CPU physics | **1.0** (success at step 2,835) | worn roller out and in the reject tray by step 707; new roller's left end found its hole at step 1,831; right end sprang into its hole at step 2,815 |
| CPU, `--variant one_end` | **0.5** | the new roller's left end in its hole, its right end left lying on the right rail |
| CPU, `--variant removed` | **0.25** | worn roller in the reject tray, new roller left in its tray |
| CPU, `--variant hold` | 0 | no arm moves |

Dream-1, run with the command above (`--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 4,500 steps
its left arm reached over the worn roller's tube and tried to lift it directly without retracting the spring-loaded hex
axle stub, so the worn roller remained captured in the rails' hex holes.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (the measured gripper poses),
never the simulator's object poses. See the docstring of `demo.py` for the sequence. In short: the right hand holds
the worn roller's tube while the left hand pulls its axle out at the left rail (the right end comes in), both hands
swing the right end up, and the right hand lifts the roller out and lays it in the reject tray. The right hand sets
the new roller on the two rails' tops (its hex settles flat, as the holes are), takes a fresh hold, feeds the axle's
left end into its hole by feel (pressing it lightly on the web and sweeping it across the hole until it slips in),
lets the right end down onto the rail and takes a fresh hold again; the left hand pulls the axle in, both hands swing
the roller level, and as the left hand eases off, the right hand sweeps the roller's end slowly across the right hole
until the spring snaps it out into the hole.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Rollers** (`roller_parts.py`): each an articulation of three bodies: the tube (root; 0.80 kg with its bearings;
  collides as a 24-sided prism), a hub on a revolute joint about the axle's axis (the bearings: drag 2e-4 N m s/rad,
  and the seals hold the axle's turn until it is twisted more than 2 degrees, about 0.017 N m), and the axle (0.35 kg)
  on a prismatic joint along its axis, travel 0 to 16 mm, its joint drive the spring (600 N/m, preloaded 4 N). The
  axle collides as a hex prism with chamfered ends (convex hulls).
- **Conveyor** (`roller_parts.py`): one fixed collider: the two rails (each web built of boxes and prisms around its
  five hex holes, so the holes are real openings), their legs, the end ties, and the four other rollers (fixed: they
  do not spin).
- **Trays**: fixed colliders.
- The scorer reads each roller's tube pose and its axle's two ends (from the axle link's pose).

## Assets

No third-party assets: the conveyor section, the rollers and the trays are this task's own geometry, built in code
(`roller_parts.py`); no meshes, scans or CAD files are used. The conveyor and rollers are generic, not copies of any
product.

## Fidelity

PhysX at 240 Hz (yam_arena's station). The rollers are articulations with a sprung axle and free-running bearings, as
described above; the four rollers that stay in place are fixed and do not spin. The axle spring is 600 N/m with a 4 N
preload (4 to 13.6 N over its 16 mm travel). Friction: steel 0.30, plastic trays 0.40; robot fingers PhysX default.
The tube collides as a 24-sided prism.
