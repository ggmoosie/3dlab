"""ak_spec.py - AK-74M real measurements (mm, frame B: bore axis, muzzle-brake face at Y=0, +Y muzzle, +Z up, +X gun left)
and the feature inventory COUNTED on the user's photos (refs_user, private). Lengths along Y come from the profile master
(ak_ref.py: left profile 163245, ~0.945 mm/px, calibrated to the 943 mm overall length); widths (X) are real-world values
to be checked on the 3/4 photos with a solved camera (S5).

Landmarks on the left profile (px, x right / row down; muzzle LEFT):
  muzzle-brake face x 154, bore row ~277.5 | butt-plate rear x 1152 | front-sight post top (250, 222)
  rear-sight notch ~(645, 226) -> sight radius 373 mm (real 379: near-orthographic photo)"""

OVERALL = 943.0                     # stock extended, brake face -> butt plate
SIGHT_RADIUS = 379.0
BARREL = 415.0

# ---- half-widths (X, mm) of the main bodies: real-world, verified later on the 3/4 photos
HW = dict(
    brake=11.25,          # 6P20 brake OD 22.5 (photo: 23 px)
    barrel_front=7.6, barrel_rear=10.5,
    fsb=10.0, fsb_ears=6.5, fsb_post=1.0,
    gas_block=11.0, gas_tube=8.6,
    hg_upper=18.0, hg_lower=24.0,
    receiver=15.2,        # stamped receiver walls
    dust_cover=15.6,
    rear_sight_block=12.0, rear_leaf=5.5,
    trigger_guard=6.0, grip=15.0, mag=13.5, stock=19.0,
    selector_standoff=2.0,
)

# ---- counted features (photo, count, notes)
FEATURES = {
    "muzzle_brake": dict(side_windows=2, front_face_step=1, face_slot=1, note="163245 window x168-187; 163339 front: stepped face, bottom slot"),
    "front_sight_block": dict(ears=2, post=1, through_hole=1, bayonet_lug=1, cleaning_rod_lug=1, note="163339 U ears + centre post; round hole (255,250)"),
    "gas_block": dict(pins=2, sling_loop=0, note="163337: two pin heads, 90 deg block"),
    "gas_tube": dict(flutes=3, vent_holes=2, front_cap=1, note="163337"),
    "handguard_retainer": dict(lever=1, note="ring at the handguard front"),
    "upper_handguard": dict(note="smooth polymer, rounded top, metal cap ring front"),
    "lower_handguard": dict(finger_ribs=6, front_vent_steps=5, note="163320 six lengthwise ribs at rear; 163337 stepped vents front"),
    "cleaning_rod": dict(count=1, note="under the barrel, into the FSB lug"),
    "receiver": dict(front_rivets=3, oval_dimple=1, small_dimples=3, fcg_pins=2, rear_trunnion_rivets=2,
                     note="163320 right side: 2 upper + 1 lower trunnion rivets, oval mag-catch dimple, 3 small dimples"),
    "dust_cover": dict(stiffening_ribs=2, spring_guide_button=1, note="long highlight lines in 163245/163254"),
    "rear_sight": dict(block=1, leaf=1, slider=1, range_marks="blank ticks only (no numbers)", note="163315/163326"),
    "side_rail": dict(plate=1, rivets=3, note="left side, 163245 x 735-835 (stamped plate + pins)"),
    "selector": dict(lever=1, finger_tab=1, detents=2, note="right side long lever, 163254"),
    "trigger_guard": dict(rivet_front=1, mag_catch=1, note="163334"),
    "pistol_grip": dict(checker_panels=2, bottom_lip=1, note="163309/163334 rectangular checkered panel each side"),
    "magazine": dict(side_rib_ticks=14, spine_rib=1, floorplate=1, note="163300/163320 short transverse ribs along the spine"),
    "stock": dict(side_recess=1, butt_trapdoor=1, sling_swivel=1, hinge_block=1, latch=1, note="163245 long rounded recess x 930-1110"),
}

# ---- photo landmarks (left profile) used for the camera/scale solve and the part partition
LM_LEFT = dict(muzzle=(154, 277.5), butt=(1152, 270), fsp_top=(250, 222), rear_notch=(645, 226),
               mag_toe=(625, 455), grip_bottom=(880, 415))
