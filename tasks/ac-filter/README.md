# Replace the mesh filter in a wall-mounted air conditioner

Wall-mounted ductless split air conditioners and fan-coil indoor units in hotels, offices, apartments, hospitals,
server rooms and commercial buildings take in return air through a washable mesh filter behind their hinged front
cover. Service manuals and preventive-maintenance schedules require pulling and cleaning or swapping this mesh filter
every 2 to 4 weeks; a clogged screen restricts airflow, frosts the evaporator coil and wastes energy. Doing it is the
same across units: lift the front cover until its friction/detent hinge holds it open, unhook the filter (lift it up
off its bottom retaining lugs, swing its bottom edge out, slide its curved top down out from behind the upper lip),
fit the clean replacement filter in reverse order (feed its curved top up behind the upper lip first, swing the lower
frame in flush against the seat, set its bottom edge down behind the three lugs), and close the cover.

**Impact on jobs:** Scheduled filter replacement and cleaning on wall-mounted split and room HVAC units is a core,
regular preventive-maintenance task for **General Maintenance and Repair Workers** (O*NET `49-9071.00`, ~1,500,000 US
workers, BLS OEWS: *"Perform routine preventive maintenance to ensure that machines continue to run smoothly, building
systems operate efficiently, or the physical condition of buildings does not deteriorate"*), **Heating, Air
Conditioning, and Refrigeration Mechanics and Installers** (O*NET `49-9021.00`, ~415,000 US workers: *"Perform
scheduled preventive maintenance, such as cleaning or replacing filters, cleaning coils, or lubricating motors"*), and
commercial/hospitality **Maids and Housekeeping Cleaners** (`37-2012.00`, ~900,000 US workers) and **Janitors and
Cleaners** (`37-2011.00`, ~2,200,000 US workers) who swap room-unit intake screens during regular room and building
maintenance rounds. Across building maintenance and HVAC service crews alone, well over 500,000 US workers perform
this specific filter swap regularly using this exact hand procedure.

The robot does it with both arms: open the cover, take the dusty filter out, fit the clean one, close the cover.

The physical skill: working a hinged cover open and shut along its arc; synchronized bimanual unhooking and blind
top-lip-then-bottom-lug re-hooking of a wide (600 mm), light, thin curved frame captured at both ends (three 6 mm lugs
below, an 11 mm lip above, with the coil 5 mm behind and the open cover above constraining the approach) that can only
be removed or seated by a coordinated lift-tilt-slide sequence without racking or twisting it; and flipping the wide
frame from lying flat on the bench to upright between two grippers.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: scene (`OBJECTS`), `success`, `score`, `distance`; docstring = what is on the table |
| `ac_parts.py` | the indoor unit (its body and its hinged cover, an articulation) and the two filters |
| `tb_parts.py` | shared shape and USD helpers, and the hook that lets yam_arena spawn the parts |
| `demo.py` | scripted two-arm demo, using only the known layout and the robot's joint readings |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, PhysX at 240 Hz). Run at 20 Hz control with object
jitter off:

    inspect-robots run --embodiment yam_arena -E task=task.py -E control_hz=20 -E object_jitter=0 \
        --policy <policy> --scorer task_score --max-steps 3000 \
        --instruction "Replace the dusty filter with the clean one: open the cover, take the dusty filter out, fit the clean filter in its place, and close the cover."

`task_score` gives 1.0 when the episode ends in success (held for 1 s), otherwise `score()` at the last step. One
fixed layout, no seeds needed. Runtime: about 30 s of simulator start-up, then about 4 control steps per second on
one GPU with the three cameras rendering (a full 3,000-step trial is about 12 minutes).

Scripted demo:

    python demo.py                      # GPU physics, as a trial runs: 1.0
    python demo.py --variant open       # the same, but the cover is left open: 0.5
    python demo.py --variant notfit     # the clean filter is left half fitted (its top behind the lip): 0.25
    python demo.py --variant outonly    # only the dusty filter is taken out: 0.25
    python demo.py --variant hold       # no arm moves: 0
    python demo.py --variant all        # all of them, one simulator session
    python demo.py --device cpu         # CPU physics, ~3x faster, for quick checks

## The scene

All positions are from the arm bases' midpoint, x forward (away from the robot), y to the robot's left.

- **Wall board**: 20 mm thick, 900 mm wide, 450 mm tall, standing across the table 0.74 m in front of the arm bases;
  the unit is mounted on it at the arms' working height, as in a service-training rig.
- **Indoor unit**, hung on the wall board, centred on the arms' midline: white plastic, 700 mm wide, 270 mm tall
  (its underside 80 mm above the table, its top 350 mm), 190 mm deep; its front 0.55 m in front of the arm bases.
  - **Outlet section** at the bottom (80 to 150 mm above the table), its top the filter's **ledge**.
  - **Front cover**: a panel 700 mm wide and 173 mm tall (from 165 to 338 mm above the table), 12 mm thick, 0.3 kg,
    hinged along its top edge (6 mm in front of the unit, 344 mm above the table), with a **finger pull** along its
    bottom edge (580 mm long, sticking out 10 mm, 5 mm tall). Its hinge is stiff: the cover stays at whatever angle
    it is left (it takes about 0.17 N m, 1 N at its bottom edge, to move it), from shut to fully open, 80 degrees
    (swung out and up, its bottom edge toward the robot), where it stops.
  - Behind the cover, the **filter's place**: the filter stands on the ledge 6 mm behind the unit's front; three
    **lugs** in front of its bottom edge (3 mm deep, 6 mm tall, 30 mm wide, at the middle and 220 mm either side) and a
    **lip** in front of its top (11 mm deep, from 300 mm above the table up to the unit's top) hold it in. Behind it,
    the coil, 5 mm from it (11 mm behind its bent top part); above it, 17 mm of room under the unit's top.
- **Filters** (two, the same shape): 600 mm wide, 168 mm tall: a flat, upright lower part 115 mm tall and a top part
  55 mm long bent back 15 degrees (following the coil); a grey plastic frame 5 mm deep with an 8 mm border and ribs,
  holding a fine mesh. Two **pull tabs** on the front of the bottom edge, 150 mm either side of the middle: each
  20 mm wide, sticking out 20 mm, 3 mm thick.
  - **Dusty filter**, 90 g, brown-grey mesh: fitted in the unit (on the ledge, behind the lugs, its top behind the
    lip).
  - **Clean filter**, 80 g, light grey mesh: lying face up on the table in front of the unit, its bottom edge (tabs
    sticking up) toward the robot, 0.36 m in front of the arm bases, centred; it rests on its bottom edge and the top
    edge of its bent part.

## Scoring

At the end of the episode:

| credit | condition |
| --- | --- |
| 0 | any part of the **dusty filter is still in the unit** (behind the unit's front, between its underside and its top) |
| 1.0 | otherwise, the **clean filter fitted** and the **cover shut**, held for 1 s (success) |
| 0.5 | otherwise, the clean filter fitted, the cover open |
| 0.25 | otherwise (the dusty filter out, the clean filter not fitted) |

- **Fitted**: the clean filter in its place: standing on the ledge (within 2 mm of it), behind the lugs (between the
  lugs and the coil: its play, 3 mm forward and 5 mm back), its top behind the lip, upright and facing out (within 3
  degrees of its fitted orientation), centred (within 10 mm across). A filter in backwards or upside down is not
  fitted.
- **Shut**: the cover within 3 degrees of closed: its bottom edge within 9 mm of the unit's front.
- Where the dusty filter ends up does not matter, as long as no part of it is in the unit.

In a real trial: a look behind the cover (or a photo) shows whether the clean filter stands on its ledge behind the
lugs with its top behind the lip; a ruler shows the cover's bottom edge within 9 mm of the unit when shut.

## Measured results

Scripted demo, 20 Hz control, object jitter 0 (run limit 3,000 steps; the demo needs about 1,800).

| run | score | notes |
| --- | --- | --- |
| GPU (`cuda:0`) | **1.0** (success at step 1,738) | cover open by step 443; dusty filter out and laid down by 885; clean filter fitted by 1,298; cover shut (1.0 degree) |
| CPU | **1.0** (1,798 steps) | the same sequence |
| CPU, `--variant open` | **0.5** | the clean filter fitted, the cover left open |
| CPU, `--variant notfit` | **0.25** | the clean filter left with its top behind the lip, its bottom out |
| CPU, `--variant outonly` | **0.25** | the dusty filter out, the clean one left on the table |
| CPU, `--variant hold` | 0 | no arm moves |

Dream-1, run with the command above (and `--policy dream -E cam_width=320 -E cam_height=176`), scored 0.0: in
3,000 steps its arms stayed near their bases, over the clean filter on the table; they never reached the unit.

## The demo

The demo uses only the task's known starting layout and the robot's own joint readings (its measured gripper
positions), never the simulator's object poses. The arms are soft: while carrying or pushing, each arm corrects its
command by its measured position error.

1. **Open the cover** (left arm, gripper closed and pointing forward, 0.25 m left of the middle): pinch the finger
   pull and swing the cover out along its arc to 35 degrees, turning the gripper with it; let go; then, from below,
   push its bottom edge up along its arc to the stop.
2. **Take the dusty filter out** (both arms, grippers pointing forward, fingers closing up and down on the tabs):
   lift it 9 mm (off the lugs), swing its bottom out 10 degrees about its top, lower it 32 mm (its top down past the
   lip), bring it out 100 mm, then turn it over about the tabs while bringing it toward the robot, and lay it face up
   on the table 0.15 m in front of the arm bases.
3. **Fit the clean filter**: pinch its tabs from above (fingers closing along the tab), turn it up about the tabs
   while lifting it, bring it to the unit tilted 8 degrees (bottom out) with its top just under the lip, raise it 28
   mm (its top up behind the lip, its bottom above the lugs), swing it upright about its top, and set it down on the
   ledge.
4. **Close the cover** (left arm): going round the cover's bottom edge, press its face down from above to 40
   degrees; going round again, push it shut with the fingertips from in front.

## How it is built

yam_arena's task files only offer single primitives and library assets. `tb_parts.py` is the task's own minimal
mechanism; nothing in yam_arena is modified: `task.py` calls `tb_parts.install()`, which wraps
`yam_arena.scene.task_object` and `object_states` when that module is imported. Parts are spawned from USD files
written to a temporary folder.

- **Unit**: an articulation: a fixed body (boxes, and a ramp for the coil behind the filter's bent part) and the
  cover on a revolute joint (0 to 80 degrees). The stiff hinge is the joint's drive, re-aimed once per control step
  at the angle the cover was left at (plus what holds it against its weight); pushed more than 1 degree from there,
  where it is held moves with it.
- **Filters**: free rigid bodies; each collides as a box (its lower part), a slanted slab (its bent top part) and
  the two tabs.

## Assets

No third-party assets: the unit and the filters are this task's own geometry, built in code (`ac_parts.py`); no
meshes, scans or CAD files are used. The unit is a generic design, not a copy of any product.

## Fidelity: what is simplified

- The filters are rigid bodies; dust is not simulated (the dusty mesh is darker and 10 g heavier).
- The cover's stiff hinge is modelled by its joint's drive (see above).
- Friction: filter and unit plastic 0.4, coil steel 0.3; robot fingers PhysX default 0.5.
