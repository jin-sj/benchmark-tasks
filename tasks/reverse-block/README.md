# Find the terminal block mounted the wrong way round and remount it the right way round

Five identical terminal blocks are latched on a DIN rail. They are not symmetric: each has a fixed hook at one end
of its base and a sprung foot at the other, and its two ends differ in shape. Four face the right way (hook toward
the far side); one was mounted the wrong way round. Neither the instruction nor the task file's docstring (which
language-model agents read) says which block it is. The robot has to find it, take it off (it cannot be pulled
straight off), turn it round, hook it back under the far lip and press it home in the same place, without
disturbing its neighbours.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `tb_parts.py` | the rail and the jointed terminal block (A3C 2.5 outline, fixed hook and sprung foot), the USD writer, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted one-arm demo, closed loop on the simulated object poses |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the blocks start latched and must start exactly on the rail):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 1200 \
        --instruction "One of the terminal blocks on the DIN rail is mounted the wrong way round. Take it off and mount it the right way round in the same place, without disturbing the other blocks."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. Runtime: about 30 s of simulator start-up, then about 3 control steps per second on one GPU with the
three cameras rendering (a full 1,200-step trial is about 6 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py                    # GPU physics, as a trial runs
    python demo.py --variant hold     # no arm moves: the block stays reversed (score 0)
    python demo.py --device cpu       # CPU physics, ~2.5x faster, for quick checks

## The scene

- **Rail:** steel TS35 x 7.5 top-hat DIN rail, 500 mm long, fixed flat to the table (as on a panel's backplate),
  open side up, running left to right, with a zinc end bracket 4.2 cm in from each end; its centre 0.14 m in front of
  the arm bases and 0.066 m right of their midline. (The reversed block sits on the arms' midline, more than 60
  degrees to the left of the right arm's base, where that arm's wrist can still turn a block a full 180 degrees.)
- **Blocks:** five identical beige polyamide terminal blocks, 7.6 g, 5.1 mm thick (along the rail), 65 mm long
  (across it) and 36 mm tall, latched 33 mm apart centre to centre (28 mm gaps), the middle one on the rail's centre.
  Each is shaped like a Weidmueller A3C 2.5: a fixed hook at one end of its base, a sprung foot at the other.
- **The reversed block:** the leftmost one, 66 mm left of the rail's centre (on the arms' midline), latched with its
  fixed hook under the near lip (toward the arms) and its sprung foot under the far lip. The other four, to its
  right, have the hook under the far lip.

## Scoring

At the end of the episode, for the reversed (leftmost) block:

| credit | condition |
| --- | --- |
| 1.0 | success: **latched the right way round in its place** and still, held for 1 s in a row, with the other four blocks undisturbed |
| 0.75 | latched the right way round elsewhere on the rail |
| 0.5 | off the rail, or on it the right way round but not latched |
| 0.25 | still the wrong way round, but no longer latched |
| 0 | as it started |

- **Latched:** its base 0.0 to 1.3 mm above the lip tops at both lip edges, within 1 mm of the rail's centreline,
  its sprung foot in (within 1 mm of rest), both seat pads pressed by the lips, level within 3 degrees.
- **The right way round:** its fixed hook toward the far side, square to the rail within 3 degrees (the wrong way
  round: within 3 degrees of 180).
- **In its place:** within 3 mm of where it started along the rail (a tenth of the gap to its neighbour).
- **Off the rail:** its base 20 mm or more above the lips, or its centre outside the rail's footprint.
- **Undisturbed:** each of the other four blocks still latched and within 1 mm of where it started along the rail.
  If one is not, success is impossible and the score is capped at 0.25.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

In a real trial these are read with a ruler and a photo: which way the block faces (the shape of its ends), whether
it sits flat and latched (a gentle tug does not lift it), and whether it and its neighbours are on their marks.

## Measured results

Scripted demo, 20 Hz control, object jitter 0, run limit 1,200 steps.

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 667) | gripped at step 129; pushed 2.4 mm toward its foot by step 167 (the command 19 mm ahead of the gripper and 5 mm below it); tipped and slid off its hook, off the rail at step 324; turned 180 degrees (0.6 degree off square) by step 394; hooked under the far lip at step 622; rotated down about the hook and latched at step 638; 0.3 mm from its place; the other four blocks did not move |
| CPU | **1.0** (success at step 783) | the same sequence; latched at step 753, 2.3 mm from its place (the limit is 3 mm); the other four blocks did not move |
| CPU, `--variant hold` | 0.0 (100 steps) | no arm moves: the block stays latched the wrong way round, the others latched |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in 1,200
steps both arms moved only a little near their start, neither came near the strip, and every block stayed as it
started.

## The demo

The script knows which block is reversed (a policy has to see it). Right arm, all in the block's own frame (its
fixed hook toward +x):

1. **Grip:** comes down over the block and grips it across its broad faces, 18 mm below its top.
2. **Take it off:** pushes it toward its sprung foot's end, the command leading (the arm is soft) and also leading
   5 mm downward (without it the block rocks onto its foot's end and the gripper slides along it), until it has moved
   2.4 mm (the fixed hook gives and the sprung foot comes out from under its lip); tips the foot's end up about the
   hook; slides it toward the hook's end until the hook is out; lifts it 35 mm.
3. **Turn it round:** 180 degrees about the vertical, in two halves.
4. **Hook it:** brings it over its place tipped 13 degrees (near end up) and 3.5 mm toward the far side, the hook's
   tooth beside the far lip at the lip's underside, correcting by the block's measured pose; slides it toward the
   arms (the command also leading 1.5 mm downward) until the tooth is under the lip, correcting its position along
   the rail as it goes.
5. **Press it home:** still holding it, rotates it down about the hook while leading the command downward, until it
   latches; holds 0.5 s; takes the downward lead out, lets go and lifts away.

The left arm does not move.

## How it is built

yam_arena's task files only offer single primitives and library assets and have no extension point for jointed
parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified: `task.py` calls
`tb_parts.install()`, which wraps `yam_arena.scene.task_object` and `object_states` when that module is imported.
Parts are spawned from USD files written to a temporary folder.

- **Block:** an articulation. The housing's look is the A3C 2.5's outline, taken from its CAD at mid-thickness; its
  collision is two boxes. The sprung foot (near end) is a slider sprung toward the rail (1 N/mm), carrying the foot's
  tooth, lead-in and sole as convex pieces. The fixed hook (far end) has a 2.5 mm tooth; it can slide outward by up
  to 2.5 mm against a 1 N/mm spring, which stands in for the plastic bending. Two small sprung seat pads press on the
  lips, as the clamp of a latched block. The reversed block is the same part, spawned turned 180 degrees.
- **Rail:** fixed, 2 mm lips.

## Fidelity: what is simplified

- The latch is a spring stand-in: the sprung foot is a 1 N/mm slider; the fixed hook's give is a 1 N/mm slide of up
  to 2.5 mm, softer than a real hook so the simulated arm can push it. Neither stiffness was measured.
- The rail's lips are 2 mm thick (1 mm on a real rail: at 240 Hz a 1 mm lip lets parts tunnel through). To keep a
  real A3C's clearance around a lip, the hook's and the foot's slots are widened in this task: a latched block's teeth
  sit 0.8 mm below the lips and its base 0.8 mm above them (0.3 mm in the other tasks built on these parts). With
  0.3 mm, sliding the hook back under the lip needed height control finer than the simulated arm has.
- The block's collision shape is simplified (two boxes, plus convex pieces for the feet); its look is the CAD outline.
- Friction: block housing polyamide 0.35, latch parts on the lips 0.25, robot fingers PhysX default 0.5.
- The block's velocity (for "still") comes from its pose change: the latch joints' armature skews PhysX's.

## Known weak points

- The reversed block is always the leftmost one; a policy that memorises the layout need not inspect the blocks.
  Changing it is a one-line change (`REVERSED` in `task.py`), but the right arm can turn a block 180 degrees only
  more than about 60 degrees to its left (its last wrist joint turns +-120 degrees): the block must stay at or left of
  the arms' midline.
- The release push is sensitive in the simulation: pushed level, the block rocks onto its foot's end and the
  gripper slides along it (seen on the GPU solver); the demo leads its push 5 mm downward to keep the hook end down.
- The rail is screwed down. With a loose rail held by the other arm, the release push failed in the simulation
  (the gripper slid along the block whenever the rail was a free body, even a heavy one that did not move), so this
  task does not test holding the rail.
- From the fixed top camera the blocks are small and the difference between their ends is hard to see; a wrist
  camera over the strip shows it.
- The blocks stand 28 mm apart, not side by side as on a full strip: the gripper's fingers need the room.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- The scorer cannot see whether the robot is still touching the block when it counts success.
