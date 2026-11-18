# Take one terminal block off a DIN rail with a screwdriver, without disturbing its neighbours

Three terminal blocks are latched on a DIN rail screwed to the table (as on a mounting plate in a cabinet). The robot
must take the middle (blue) block off and put it in a tray, leaving the two beige blocks on either side latched where
they are.

A latched block can't be pulled or pushed off. As with real DIN-rail terminal blocks, it comes off with a flat-blade
screwdriver in its release slot:
1. put the blade into the release slot at the block's near end (toward the arms);
2. lever the sprung foot outward, so its tooth comes out from under the rail's lip;
3. tip the near end up a few degrees;
4. slide the block a few millimetres toward the far side, so its rigid fixed hook comes out from under the far lip;
5. lift it off.

The screwdriver stands in a holder beside the rail. The usual way is two-handed, and so is the demo: the left arm
holds the foot out with the screwdriver while the right arm tips the block and takes it off.

## Changes since the last review

| review point | what changed |
| --- | --- |
| **Construct validity:** the no-screwdriver variant earned credit by shifting the still-latched block | `no_lever` now scores **0**: its sprung foot never moves more than 0.1 mm out past where it sits latched (measured against the rail), and the block's 2.9 mm slide carries the foot back in under its lip. |
| **Scoring:** the 0.25 tier rewarded small shifts or tilts | 0.25 now needs a **measurable release of the sprung foot**: its tooth 2.0 mm out past its latched position, measured against the rail (clear of the lip's edge), or the near end lifted 6.5 mm (the foot above the lip). A shifted or tipped block still latched scores 0. A new variant, `lever_only`, shows the tier: the screwdriver levers the foot 2.45 mm out and holds it there (0.25). |
| **Clarity and reproducibility:** the README's 1.6 mm tab and slot coordinates disagreed with the 3.0 mm tab and 2.4 mm slot that are implemented | *The latch* now gives the implemented geometry: cheek x -30.0 to -27.8 mm, slot 2.4 mm (x -32.4 to -30.0 mm) by 4.8 mm, tab 3.0 mm (x -35.4 to -32.4 mm). The blade is given as 3.5 x 1.0 mm everywhere. It fits the slot with 1.4 mm of play across and 1.3 mm along, and it is put in 0.2 mm from the slot's middle in every screwdriver run. |
| **Rerun the scoring demonstrations** | All variants were rerun with the new scoring, and a `lifted` variant now covers the 0.5 tier: `full` 1.0, `held` 0.75, `lifted` 0.5, `lever_only` 0.25, `no_lever` 0. Dream-1 stays at 0.0: its block stayed latched, and the stricter tier can't raise a score. |

## Changes after the first review

| review point | what changed |
| --- | --- |
| **Construct validity:** the removal depended on 2.5 mm of fixed-hook deflection, not on how the work is done | The fixed hook is now **rigid** (no deflection at all). The block is released the way DIN-rail terminal blocks are normally removed: a flat screwdriver in the **release slot** at the sprung foot levers the foot out, then the block is tipped and slid off its fixed hook. The old push no longer releases anything: the `no_lever` variant (the same arm motions without the screwdriver) leaves the block on the rail. |
| **Clarity and reproducibility:** latch geometry and release force not specified | New section *The latch*: every latch dimension (hook and foot reach, foot travel, slot size and position, tab), the foot's spring (1.0 N/mm) and the **release force (2.0 N at the lip's edge)**, specified as a buildable equivalent of this task's own design. |
| **Include any tool required for normal removal** | A **flat-blade screwdriver** (3.5 mm blade) in a holder is now part of the scene; the instruction names it. |
| **Update setup, fidelity note and demonstration** | Setup, scoring notes, fidelity note and `task.json` describe the new mechanism. The rail is now screwed down (as on a mounting plate), so one arm uses the screwdriver while the other takes the block off. New demo (1.0), new variants (`held` 0.75, `no_lever`), new Dream-1 run (0.0). The old push-based benches and runs were removed. |

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `tb_parts.py` | the rail, the tray, the screwdriver and its holder, and the jointed terminal block (this task's own generic push-in block: rigid fixed hook, sprung foot with a release slot), the USD writer, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted two-arm demo, closed loop on the simulated object poses, and its variants |
| `runs/full/` | the full demo run: `result.json`, `setup.png`; its video (the three cameras in one) is the task's uploaded demo video |
| `runs/held/`, `runs/lifted/`, `runs/lever_only/`, `runs/no_lever/` | the other variants' `result.json` |
| `runs/dream1/` | the Dream-1 run: `eval_log.json.gz`; its video is the task's uploaded model-run video |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off (the blocks start latched and must start exactly on the rail):

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 -E device=cpu \
        --policy <policy> --scorer task_score --max-steps 2400 \
        --instruction "Take the blue terminal block off the DIN rail with the screwdriver and put it in the tray, and leave the two beige blocks where they are."

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step. One fixed layout, no
seeds needed. The demo was run on CPU physics (`-E device=cpu`); the cameras render on the GPU. Runtime: about 30 s
of simulator start-up, then about 10 control steps per second on CPU PhysX with the three cameras rendering (a full
2,400-step trial is about 4-5 minutes).

Scripted demo (drives the same embodiment and task-file scoring directly):

    python demo.py --device cpu                       # result.json, steps.jsonl, one video per camera
    python demo.py --device cpu --variant held        # the block is lowered onto the tray floor and held (0.75)
    python demo.py --device cpu --variant lifted      # the block lifted off the rail and held above it (0.5)
    python demo.py --device cpu --variant no_lever    # the same arm motions, but without the screwdriver (0)
    python demo.py --device cpu --variant lever_only  # only the screwdriver: the foot levered out and held (0.25)
    python demo.py --device cpu --no-video

## The scene

- **Rail** (fixed): steel TS35 x 7.5 top-hat DIN rail, 500 mm long, with two zinc end brackets (4.2 cm in from each
  end), screwed flat to the table, open side up, running left to right. Its centre is 0.30 m in front of the arm
  bases and 0.04 m right of their midline.
- **Blocks:** three terminal blocks latched on the rail, 33 mm apart centre to centre (28 mm gaps): beige, blue,
  beige from the left; the blue one 2 cm right of the rail's centre. Each is polyamide, 7.6 g, 5.1 mm thick (along
  the rail), 65 mm long (across it) and 36 mm tall (29 mm above the rail). The fixed hook is at the far end (away
  from the arms), the sprung foot and its release slot at the near end. See *The latch*.
- **Screwdriver and holder:** a flat-blade electrician's screwdriver (blade 3.5 mm wide, 1.0 mm thick, 10 mm long; 3 mm shaft;
  20 mm square handle 45 mm long; 75 mm overall; 60 g), standing handle up in a black plastic holder (40 x 40 mm,
  50 mm tall, a 9 mm hole). The holder's centre is 0.20 m in front of the arm bases and 0.08 m left of their
  midline.
- **Tray:** an open, shallow black plastic tray, inside 170 x 64 mm, walls 3 mm thick and 12 mm high, its centre
  0.18 m in front of the arm bases and 0.22 m right of their midline (in front of the right arm, nearer the arms than
  the rail), empty.

## The latch

The block is this task's own design, a physically reproducible equivalent of a push-in terminal block's latch. All
dimensions are in the block's frame: across the rail from its centreline (toward the arms negative), up from the lip
tops.

| feature | dimensions |
| --- | --- |
| fixed hook (far end) | rigid, part of the housing; its tooth reaches 2.5 mm in under the far lip, 0.3 mm below the lip's underside |
| sprung foot (near end) | slides outward (toward the arms) on a straight guide, up to 3.5 mm; its tooth reaches 2.0 mm in under the near lip, so it clears the lip's edge after 2.0 mm |
| foot spring | 1.0 N/mm, preloaded to 0 at rest: **the release force is 2.0 N at 2.0 mm (clear of the lip), 2.4 N at the demo's 2.4 mm**, pushed outward at the foot |
| release slot | at the near end, open at the top: 2.4 mm wide (across the rail, x -32.4 to -30.0 mm) x 4.8 mm (along it, the foot's width), 9 mm deep, between a fixed cheek on the housing (x -30.0 to -27.8 mm, 2.2 mm thick) and an L-shaped tab on the foot (an upright 3.0 mm thick, x -35.4 to -32.4 mm, and a 1 mm floor under the slot); all three move out with the foot except the cheek |
| how a screwdriver opens it | a 3.5 mm flat blade, 0.5-1.0 mm thick, goes in the slot from above, its 3.5 mm width along the rail: with the simulation's 1.0 mm blade that leaves 1.4 mm of play across and 1.3 mm along. Levering or pushing it outward pushes the tab, and with it the foot, out |
| seat pads | two pads on the base press down on the lips with 3 N/mm (the clamp of a latched block) |

The rail's lips are 17.5 mm from its centreline at their outer edges.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 1.0 | success: the blue block **released and resting on the tray's floor**, still, held for 1 s in a row, with both beige blocks undisturbed |
| 0.75 | the blue block in the tray (its centre inside the tray's inner walls, its lowest point below their tops), released or still held |
| 0.5 | the blue block **off the rail**: its base 20 mm or more above the lips, or its centre outside the rail's footprint |
| 0.25 | the blue block's **sprung foot released**: its tooth 2.0 mm or more out past its latched position, measured against the rail (so clear of the lip's edge), or the block's near end lifted 6.5 mm or more off the lip (the foot's lowest point above the lip's top). A block that is shifted or tipped but still has its foot under the lip scores 0 |
| 0 | otherwise |

- **Undisturbed:** each beige block still latched (base within -0.5..+0.8 mm of the lip tops at both lip edges,
  centred within 1 mm, feet in, square and level within 3 degrees) and within 1 mm of where it started along the rail.
- If a beige block was disturbed, success is impossible and the score is capped at 0.25. (The rail is screwed down;
  the scorer still checks it moved less than 10 mm and turned less than 3 degrees.)
- **Released and resting on the tray's floor:** no part of the robot touching the block (the simulation reads the
  robot's contact force on the block's housing, which must stay under 0.05 N), and the block's lowest point within
  2 mm of the tray floor's top, so the tray, not the gripper, holds it up. Both must hold for the whole 1 s.
- **Still:** under 10 mm/s and 0.5 rad/s, from the block's pose change over the last control step.

In a real trial these are read with a ruler, calipers and a photo: whether the blue block lies in the tray with
the gripper open and clear of it (no contact) or is off the rail; for 0.25, the foot's outer face against a mark on
the rail made where it sat latched (2.0 mm out) or the near end's height above the lip (6.5 mm); whether the beige
blocks have moved along the rail (marks on the rail) or come loose.

No unintended strategy scores: the latched block can't be pulled, pushed or levered off the rail without opening its
sprung foot (its fixed hook is rigid), and the foot can only be opened through the release slot. Prying the beige
blocks or pulling the block off by force disturbs them or leaves it latched (capped at 0.25 or 0).

## Measured results

Scripted demo, CPU physics (`--device cpu`), 20 Hz control, object jitter 0, run limit 2,400 steps.

| variant | score | notes |
| --- | --- | --- |
| `full` | **1.0** (success at step 933) | the screwdriver picked from its holder; the blade lined up over the release slot (within 0.5 mm) and put in (0.2 mm from the slot's middle, 1.4 mm above its floor) by step 375; levered: the foot out 1.1 mm by step 475; the right hand grips the block's far end and tips it while the blade holds the foot out (the foot out up to 3.5 mm); slid 3.8 mm toward the far side (the hook out) by step 796; off the rail at step 826; released on the tray floor with no robot contact; the beige blocks undisturbed |
| `held` | **0.75** | the same until the tray; the block is lowered onto the tray floor and held there to the end: in the tray, never released |
| `lifted` | **0.5** | the same as `full` until the block is lifted off the rail (step 826); then held 41 mm above the lips to the end, never carried to the tray |
| `lever_only` | **0.25** | only the screwdriver: picked, put in the release slot, levered until the foot's tooth is 2.4 mm out past its latched position (by step 522, the command leading by 22 mm), then held there to the end (2.45 mm out); the right arm stays where it starts; the beige blocks undisturbed |
| `no_lever` | **0** | the same arm motions as `full` without the screwdriver (it stays in its holder): the right hand tips and slides the block 2.9 mm toward the far side, but the sprung foot never comes more than 0.1 mm out past its latched position and the block stays on the rail |

`no_lever` shows the release really needs the screwdriver: without it the block doesn't come off and nothing scores.
Each variant's `result.json` is in `runs/<variant>/`.

The arms are soft in the simulation: the lever and the slide each need the command to lead the hand by 15-31 mm.

Dream-1, run with the command above (`--policy dream -E cam_width=320 -E cam_height=176`, 2,400 steps, 176 s), scored 0.0: both arms moved toward the screwdriver and over the rail, but the screwdriver stayed in its holder and the blue block stayed latched.

## The demo

Left arm (the screwdriver):
1. takes the screwdriver from its holder by the handle (fingers closing across the rail) and lifts it out;
2. leans it 6 degrees away from the block (handle toward the arms), so the two hands don't meet over the block;
3. lines the blade's tip up over the release slot by its measured position (correct, settle, measure), puts it in;
4. levers: pushes the blade toward the arms, the command leading, which pushes the foot's tab and the foot out;
5. holds the foot out while the right arm tips the block (following the near end up), then pulls the blade out and
   moves away.

Right arm (the block), once the foot is out:
1. comes down over the block's far end, leaning 20 degrees so its body stays beyond the block, and grips the block
   across its broad faces 30 mm from its middle (clear of the screwdriver);
2. tips the near end up along an arc about the hook;
3. once the blade is out, slides the block toward the far side until the hook's tooth is out from under its lip;
4. lifts it clear, carries it to the tray and lets go just above the floor; moves away.

## How it is built

yam_arena's task files only offer single primitives and library assets and have no extension point for jointed
parts. `tb_parts.py` is the task's own minimal mechanism; nothing in yam_arena is modified: `task.py` calls
`tb_parts.install()`, which wraps `yam_arena.scene.task_object` and `object_states` when that module is imported.
Parts are spawned from USD files written to a temporary folder.

- **Block:** an articulation, the geometry in *The latch*.
  - The housing's look is a generic push-in terminal block outline of this task's own design; its collision is two
    boxes.
  - The fixed hook is part of the housing (rigid).
  - The sprung foot (near end) is its own link on a prismatic joint across the rail, sprung inward (1 N/mm, up to
    3.5 mm out). It carries the foot's tooth and sole and the release slot's tab as convex pieces.
  - Two small sprung seat pads press on the lips, as the clamp of a latched block.
- **Rail:** fixed to the table (screwed down).
- **Screwdriver:** a free rigid body; **holder:** fixed.

## Assets

No third-party assets: the rail, the tray, the screwdriver and its holder and the terminal blocks are this task's own geometry, built in code
(`tb_parts.py`); no meshes, scans or CAD files are used.

## Fidelity: what is simplified

- **The latch:** the sprung foot's plastic flexure is a 1 N/mm slider (release 2.0 N at the lip's edge). This is
  this task's specification for a physically built equivalent (e.g. a 3D-printed block with a 1 N/mm spring on the
  foot), not a measured value of a commercial block. The fixed hook is rigid.
- **The blade** is 3.5 x 1.0 mm (a 0.7 mm blade tunnelled through the foot's tab in PhysX), and the tab is 3.0 mm
  thick (a 1.6 mm tab let the blade tunnel through). The 2.4 mm slot takes any 3.5 mm blade 0.5-1.0 mm thick, and
  leaves the soft simulated arm room to place it.
- **The block's collision shape** is simplified (two boxes, plus convex pieces for the feet and the slot); its look is
  a generic outline, not a particular product's.
- **The rail's lips** are 2 mm thick (1 mm on a real rail): at 240 Hz a 1 mm lip lets parts tunnel through.
- **Friction:** rail steel 0.3, block housing polyamide 0.35, latch parts on the lips 0.25, screwdriver handle 0.9
  (rubber), the table and robot fingers PhysX default 0.5.
- **The block's velocity** (for "still") comes from its pose change: the latch joints' armature skews PhysX's.

## Known weak points

- Both hands work at the same small block: the left hand's screwdriver at the near end, the right hand at the far
  end. The demo leans both hands apart; upright, they met over the block.
- `tb_parts.install()` wraps two functions of `yam_arena.scene`; a yam_arena update that changes them stops the task
  with an error.
- Object jitter must be 0: the blocks start latched.
