"""Cortlandt Station: the single source of truth for the level build (level-factory stage 2).
Layout v3 "The Arcade" numbers come from spec_v3.py (metres, spec frame: x west->east along the platform, y south->north,
z up, platform top = 0). This module adds every fixture list the layout, the plan page and the UE build read, and the
conversion to the level-factory UE frame (cm; X from the entrance to the back = east, Y right when facing +X = south, Z up).
Pure data + helpers, no bpy."""
import math
from spec_v3 import *          # noqa: F401,F403  (every v3 number)
import spec_v3 as V


def ue(x, y, z=0.0):
    """spec metres -> UE cm (X east, Y south, Z up)"""
    return (round(x * 100, 1), round(-y * 100, 1), round(z * 100, 1))


def ue_rect(r, z0=0.0, z1=0.0):
    x0, y0, x1, y1 = r
    return [round(x0 * 100, 1), round(x1 * 100, 1), round(-y1 * 100, 1), round(-y0 * 100, 1), round(z0 * 100, 1), round(z1 * 100, 1)]


def stair_run(risers):
    return (risers - 2) * V.TREAD + V.LANDING


def stair_rect(st, risers):
    """footprint (x0, y0, x1, y1) of a straight stair from its top nosing along dir"""
    run = stair_run(risers)
    (x, y), (dx, dy), w = st["top"], st["dir"], st["width"]
    xa, xb = sorted((x, x + dx * run))
    return (xa, y - w / 2, xb, y + w / 2) if dx else (x - w / 2, min(y, y + dy * run), x + w / 2, max(y, y + dy * run))


def _in(r, x, y, pad=0.0):
    return r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad


PLATFORM_STAIR_RECTS = [stair_rect(s, V.PLATFORM_STAIR_RISERS) for s in V.PLATFORM_STAIRS]
STREET_STAIR_RECTS = [stair_rect(s, V.STREET_STAIR_RISERS) for s in V.STREET_STAIRS]

# --- platform level ---------------------------------------------------------------------
PLATFORM_COLUMNS = [(round(-34.29 + i * V.COLUMN_PITCH, 3), sy * V.COLUMN_ROW_Y)
                    for i in range(16) for sy in (-1, 1)]
PLATFORM_LIGHTS = [dict(kind="fluorescent_strip", x=round(-32.0 + i * 4.572, 2), y=0.0, z=V.PLATFORM_CEIL - 0.1, lumens=5200)
                   for i in range(15)]
TRAINS = []
for tr, track_y in ((V.TRAIN_A, -6.75), (V.TRAIN_B, 6.75)):
    for c in range(tr["cars"]):
        x0 = tr["start_x"] + c * (V.TRAIN_CAR["length"] + V.TRAIN_CAR["gap"])
        TRAINS.append(dict(kit=tr["kit"], rect=(x0, track_y - V.TRAIN_CAR["width"] / 2, x0 + V.TRAIN_CAR["length"], track_y + V.TRAIN_CAR["width"] / 2)))
RANGE_TARGETS = [dict(dist=d, x=35.0 + d, y=6.75) for d in (10, 25, 50) if 35.0 + d <= 85.0]

# --- mezzanine --------------------------------------------------------------------------
CONCOURSE_COLUMNS = [(x, sy * V.CONCOURSE_COLUMNS_Y) for x in range(-26, 34, 6) for sy in (-1, 1)
                     if not any(_in(r, x, sy * V.CONCOURSE_COLUMNS_Y, 0.4) for r in PLATFORM_STAIR_RECTS)]
MEZZ_LIGHTS = [dict(kind="light_box", x=float(x), y=0.0, z=V.MEZZ_CEIL - 0.05, lumens=6000)
               for x in range(-25, 34, 6) if not any(_in(r, x, 0.0, 0.5) for r in PLATFORM_STAIR_RECTS)]
ROOM_HERO_LIGHTS = {k: dict(x=(r["rect"][0] + r["rect"][2]) / 2, y=(r["rect"][1] + r["rect"][3]) / 2, ident=r["light"])
                    for k, r in V.MEZZ_ROOMS.items()}
SHOPFRONTS = [dict(room=k, y=(4.0 if r["rect"][1] >= 4.0 else -4.0), x0=r["rect"][0], x1=r["rect"][2])
              for k, r in V.MEZZ_ROOMS.items() if r["rect"][0] >= V.FARE_LINE_X and r["rect"][2] <= 34.0]
TURNSTILES = [(V.FARE_LINE_X, -3.0 + i * 1.0) for i in range(7)]
RAID_BOARD = dict(x0=-2.0, x1=6.0, y=12.0, note="north wall of the Node, faces south")
NODE_SERVICES = ["trade_post", "infirmary", "gunsmith", "stash"]

# --- street -----------------------------------------------------------------------------
def street_blocks():
    """building lots (rects) between the streets, 15 m deep behind the sidewalks"""
    ys = [(V.AVENUE_HALF + V.SIDEWALK_AVE, V.AVENUE_HALF + V.SIDEWALK_AVE + V.LOT_DEPTH),
          (-(V.AVENUE_HALF + V.SIDEWALK_AVE + V.LOT_DEPTH), -(V.AVENUE_HALF + V.SIDEWALK_AVE))]
    xs = [(-95.0, -60.0 - V.CROSS_HALF - V.SIDEWALK_CROSS), (-60.0 + V.CROSS_HALF + V.SIDEWALK_CROSS, 60.0 - V.CROSS_HALF - V.SIDEWALK_CROSS),
          (60.0 + V.CROSS_HALF + V.SIDEWALK_CROSS, 95.0)]
    out = []
    for xa, xb in xs:
        for ya, yb in ys:
            r = (xa, ya, xb, yb)
            if V.STREET_ZONES["plaza"]["rect"][0] < xb and V.STREET_ZONES["plaza"]["rect"][2] > xa and ya > 0:
                pl = V.STREET_ZONES["plaza"]["rect"]
                if xa < pl[0]:
                    out.append((xa, ya, pl[0], yb))
                if pl[2] < xb:
                    out.append((pl[2], ya, xb, yb))
                continue
            out.append(r)
    return out


STREET_BLOCKS = street_blocks()
ZOMBIE_SPAWNERS = [(round(97.0 * math.cos(a), 1), round(60.0 * math.sin(a), 1)) for a in
                   [2 * math.pi * (i + 0.5) / V.ZOMBIE_SPAWNERS for i in range(V.ZOMBIE_SPAWNERS)]]
TIDE_GAPS = [(-95.0, 0.0), (95.0, 0.0), (0.0, 58.0), (0.0, -58.0)]
TUTORIAL_PATH = [(-90.0, 0.0), (-60.0, 0.0), (-60.0, 8.0), (-46.0, 9.5), (-42.0, 8.5)]

# --- budgets (level-factory GunRange numbers) -------------------------------------------
BUDGETS = dict(map_tris=500_000, unique_meshes=70, material_instances=30, kit_prop_tris=(200, 8000),
               lights_per_floor_px_p95=4, shadow_casters=13, ev100=6.0)

DECISIONS = [
    ("Shop bay width", "Bays are 12 m wide (8 m deep). Four players fit at the trader counter with about 1 m each. Keep 12 m, or go to 14 m and shorten the outer bays?"),
    ("Black market access", "Behind the NO ENTRY staff door. Locked until a quest unlocks it, or always open?"),
    ("Red entrance", "The south (red globe) stair is welded shut as a landmark. Should it ever open (event, late-game shortcut), or stay shut?"),
    ("Street interiors", "18 enterable shops out of 84 fronts. Build all 18 in the first pass, or 6 hero interiors first (drugstore, bodega, deli, pawn shop, laundromat, walk-up lobby)?"),
    ("Test range", "Down the Track B tunnel at 10 / 25 / 50 m, reusing the GunRange target kit. OK to share that kit, or a station-themed target set?"),
]
