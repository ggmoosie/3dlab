"""iface.py - the AK-74M's modular contract as DATA (B frame, mm: bore axis z=0, muzzle face y=0, gun at y<0, +X left).
SLOTS     what can be swapped/removed, what it hangs off (parent slot + socket), and the envelope a replacement must fit.
SOCKETS   named attach frames (pos, axis) shared by the base modules and any aftermarket module (Zenit B-10/B-11/B-33,
          other brakes, railed covers): a new handguard is a new module file that fills the same slot + envelope.
Positions come from the profile master (ak_ref, ~0.945 mm/px); internal ones (bolt face, carrier) are real-world layout."""
import ak_geo as G

by, bz = G.by, G.bz
Y = dict(
    brake_rear=by(241), thread_front=-62.0, thread_rear=by(244),
    fsb_front=by(238), fsb_rear=by(279), gasblock_front=by(338), gasblock_rear=by(375),
    gastube_front=by(352), hg_cap_front=by(444), hg_cap_rear=by(453), rsight_front=by(564), rsight_rear=by(622),
    trunnion_front=by(622), bolt_face=-470.0, magwell_front=by(627), magwell_rear=by(731),
    trigger_slot_front=by(781), trigger_slot_rear=by(813), rear_trunnion_front=-668.0, receiver_rear=by(892),
    port_front=by(640), port_rear=by(716), stock_front=by(892), butt=by(1152),
)
Z = dict(
    piston=26.5, gastube_r=9.4, rail_top=bz(258), cover_bottom=bz(262), cover_top=bz(232), receiver_floor=bz(303),
    handle=21.0, guide=27.5, guide_button=30.0,
)
REC_T = 1.2                          # stamped receiver wall (mm)
COVER_T = 0.9                        # dust cover sheet
POLY_T = 2.6                         # polymer handguard wall
CARRIER_TRAVEL = 130.0               # bolt carrier stroke, for the reload / fire animation

SOCKETS = {                          # name: (x, y, z), axis
    "muzzle_thread":   ((0, Y["thread_front"], 0), "+y"),
    "fsb_seat":        ((0, Y["fsb_rear"], 0), "+y"),
    "gas_port":        ((0, (Y["gasblock_front"] + Y["gasblock_rear"]) / 2, Z["piston"]), "+z"),
    "hg_cap":          ((0, Y["hg_cap_rear"], 0), "-y"),
    "hg_rear_plate":   ((0, Y["trunnion_front"], 0), "+y"),
    "gastube_lever":   ((-G.HW["rear_sight_block"], Y["rsight_front"] - 6, Z["piston"]), "-x"),
    "cover_front":     ((0, Y["rsight_rear"], Z["rail_top"]), "+y"),
    "cover_latch":     ((0, Y["receiver_rear"] + 6, Z["guide_button"]), "+z"),
    "side_rail":       ((G.HW["receiver"] + 2.2, G.by(785), G.bz(294)), "+x"),
    "carrier_rails":   ((0, Y["bolt_face"], Z["rail_top"] - 3), "-y"),
    "charging_handle": ((-G.HW["receiver"] - 12, by(633), Z["handle"]), "-x"),
    "magwell":         ((0, (Y["magwell_front"] + Y["magwell_rear"]) / 2, Z["receiver_floor"]), "-z"),
    "grip_mount":      ((0, G.by(850), G.bz(318)), "-z"),
    "stock_hinge":     ((G.HW["receiver"], Y["receiver_rear"], G.bz(289.5)), "+x"),
}

SLOTS = {   # slot: parent, socket, removable, envelope (what a replacement must stay inside / clear)
    "muzzle":        dict(parent="barrel", socket="muzzle_thread", removable=True,
                          envelope=dict(thread_r=7.0, thread_len=Y["thread_front"] - Y["thread_rear"])),
    "barrel":        dict(parent="receiver", socket="hg_rear_plate", removable=False),
    "front_sight":   dict(parent="barrel", socket="fsb_seat", removable=False),
    "gas_block":     dict(parent="barrel", socket="gas_port", removable=False),
    "cleaning_rod":  dict(parent="front_sight", socket="fsb_seat", removable=True),
    "upper_handguard": dict(parent="rear_sight", socket="gastube_lever", removable=True,
                          envelope=dict(y=(Y["gastube_front"], Y["rsight_front"]), tube_axis_z=Z["piston"], tube_r=Z["gastube_r"],
                                        max_hw=20.0, note="Zenit B-10 class: rail top, same tube + lever")),
    "lower_handguard": dict(parent="barrel", socket="hg_cap", removable=True,
                          envelope=dict(y=(Y["hg_cap_rear"], Y["trunnion_front"]), barrel_clear_r=11.5, max_hw=26.0,
                                        note="Zenit B-11/B-30 class: drops in between cap and rear plate")),
    "rear_sight":    dict(parent="receiver", socket="cover_front", removable=False),
    "dust_cover":    dict(parent="receiver", socket="cover_latch", removable=True,
                          envelope=dict(y=(Y["rsight_rear"], Y["receiver_rear"]), inner_hw=G.HW["receiver"] + 0.1,
                                        rail_z=Z["rail_top"], note="Zenit B-33 class: same latch + rear sight notch")),
    "bcg":           dict(parent="receiver", socket="carrier_rails", removable=True, travel=CARRIER_TRAVEL),
    "recoil":        dict(parent="receiver", socket="cover_latch", removable=True),
    "fcg":           dict(parent="receiver", socket="grip_mount", removable=False),
    "selector":      dict(parent="receiver", socket="charging_handle", removable=False, anim="rotate safe/auto/semi"),
    "grip":          dict(parent="receiver", socket="grip_mount", removable=True),
    "stock":         dict(parent="receiver", socket="stock_hinge", removable=True, anim="fold left"),
    "magazine":      dict(parent="receiver", socket="magwell", removable=True),
    "side_rail":     dict(parent="receiver", socket="side_rail", removable=False, note="optic mount"),
    "receiver":      dict(parent=None, socket=None, removable=False),
}
