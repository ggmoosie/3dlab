"""Every number and registry for Cortlandt Station (layout v3: "The Arcade").
Pure data — no bpy, no logic. Units: metres. X along the platform (west→east), Y across
(south→north), Z up. Platform top = 0. Rects are (x0, y0, x1, y1)."""

# --- vertical stack -----------------------------------------------------------------
Z_PLATFORM = 0.0
Z_MEZZ = 5.2
Z_STREET = 10.0          # sidewalk level; roadway 0.15 lower
ROAD_DROP = 0.15
PLATFORM_CEIL = 4.3
MEZZ_CEIL = 8.8

TREAD = 0.28             # NYC ~11 in
LANDING = 1.5
STREET_STAIR_RISERS = 27         # 4.8 m / 27 = 17.8 cm
PLATFORM_STAIR_RISERS = 29       # 5.2 m / 29 = 17.9 cm

# --- platform level -----------------------------------------------------------------
PLATFORM = (-35.0, -4.5, 35.0, 4.5)
TRACK_A = (-85.0, -9.0, 85.0, -4.5)      # south: the live train
TRACK_B = (-85.0, 4.5, 85.0, 9.0)        # north: derelicts + test range
TRACKBED_Z = -1.1
STATION_HALF_LEN = 45.0
TUNNEL_HALF_LEN = 85.0
COLUMN_PITCH = 4.572             # 15 ft
COLUMN_ROW_Y = 3.3
TRAIN_CAR = dict(length=15.5, width=2.7, gap=0.9, doors_per_side=4, door_w=1.27)
TRAIN_A = dict(cars=4, start_x=-31.0, kit="SM_TrainCar_R142")
TRAIN_B = dict(cars=3, start_x=-33.0, kit="SM_TrainCar_Derelict")
PLATFORM_ZONES = {
    "boarding":     dict(rect=(6.0, -4.5, 24.0, 4.5), note="between the two stair feet"),
    "route_board":  dict(rect=(-35.0, -4.5, -29.0, 0.0), note="pick a stop at the lead car cab"),
    "test_range":   dict(rect=(35.0, 4.5, 80.0, 9.0), note="Track B tunnel, 10/25/50 m targets"),
    "range_steps":  dict(rect=(31.0, 2.5, 35.0, 4.5), note="steps down to Track B trackbed"),
}

# --- mezzanine: a straight Turnstyle-style arcade directly above the platform -----------
MEZZ = (-42.0, -12.0, 42.0, 12.0)
FARE_LINE_X = -28.0
CONCOURSE = (-28.0, -4.0, 34.0, 4.0)     # 8 m wide, glass shopfronts both sides
CONCOURSE_COLUMNS_Y = 3.4                # white riveted I-beams just inside the shopfronts
NODE = (-4.0, -12.0, 8.0, 12.0)          # the crossroads: main stair, raid board, compass rose
COMPASS_ROSE = dict(center=(2.0, -7.8), r=3.5)
CENTER_KIOSKS = [(-24.0 + i * 4.5, 0.0) for i in range(4)]   # glass display islands = the market
PLATFORM_STAIRS = [              # floor openings in the concourse with railings. top nosing at mezz
    dict(name="Stair_Main", top=(-3.0, 0.0), dir=(1, 0),  width=3.0),
    dict(name="Stair_East", top=(33.0, 0.0), dir=(-1, 0), width=2.4),
]
STREET_STAIRS = [                # sidewalk stairs against the building line, descend east into the Gate
    dict(name="Kiosk_North", top=(-42.0, 8.5),  dir=(1, 0), width=2.0,
         globe="green", state="open", note="main entrance beside the corner newsstand"),
    dict(name="Kiosk_South", top=(-42.0, -8.5), dir=(1, 0), width=2.0,
         globe="red", state="welded", note="sealed, visible from the Gate"),
]
# Rooms: rect, pre-outbreak use, current use, light identity.
# Most-used services hug the Node (main stair); outer bays are less frequent stops.
MEZZ_ROOMS = {
    "gate":         dict(rect=(-42.0, -12.0, -28.0, 12.0), was="Unpaid area",    light="sodium orange, CCTV glow"),
    "watch_post":   dict(rect=(-37.0, -2.0, -33.0, 2.0),   was="Token booth",    light="red alarm lamp"),
    "mechanic":     dict(rect=(-28.0, 4.0, -16.0, 12.0),   was="Locksmith",      light="orange caged bulbs"),
    "trade_post":   dict(rect=(-16.0, 4.0, -4.0, 12.0),    was="Newsstand",      light="warm marquee bulbs"),
    "gunsmith":     dict(rect=(8.0, 4.0, 20.0, 12.0),      was="Shoe repair",    light="red neon"),
    "workshop":     dict(rect=(20.0, 4.0, 34.0, 12.0),     was="Phone repair",   light="cool white + printer glow"),
    "galley":       dict(rect=(-28.0, -12.0, -16.0, -4.0), was="Pizza counter",  light="string lights"),
    "infirmary":    dict(rect=(-16.0, -12.0, -4.0, -4.0),  was="Pharmacy",       light="clinical white"),
    "stash":        dict(rect=(8.0, -12.0, 20.0, -4.0),    was="Lost & found",   light="cool fluorescent"),
    "bunks":        dict(rect=(20.0, -12.0, 34.0, -4.0),   was="Crew storage",   light="dim red night lamps"),
    "black_market": dict(rect=(34.0, 4.0, 42.0, 12.0),     was="Signal room",    light="UV purple"),
    "staff_hall":   dict(rect=(34.0, -4.0, 42.0, 4.0),     was="Staff corridor", light="flicker"),
    "generator":    dict(rect=(34.0, -12.0, 42.0, -4.0),   was="Fan plant",      light="hazard amber"),
}

# --- street -------------------------------------------------------------------------
AVENUE_HALF = 6.0                # Greenwich St, curb to curb 12 m (Breach fits it exactly)
SIDEWALK_AVE = 4.5
CROSS_STREETS = [dict(name="Cortlandt St", x=-60.0), dict(name="Dey St", x=60.0)]
CROSS_HALF = 5.0
SIDEWALK_CROSS = 4.0
STREET_BOUNDS = (-100.0, -62.0, 100.0, 62.0)
BARRICADE = (-95.0, -58.0, 95.0, 58.0)
LOT_DEPTH = 15.0
STREET_ZONES = {
    "spawn":      dict(rect=(-95.0, -6.0, -85.0, 6.0),  note="overrun NYPD checkpoint, tutorial start"),
    "pharmacy":   dict(rect=(-51.0, 10.5, -41.0, 25.5), note="Cortlandt Drugs, first loot, next to green kiosk"),
    "plaza":      dict(rect=(15.0, 10.5, 51.0, 45.0),   note="pocket park, overrun FEMA tent camp"),
    "walkups":    dict(rects=[(-79.0, 25.0, -69.0, 33.0), (69.0, 20.0, 79.0, 28.0), (69.0, -40.0, 79.0, -32.0)]),
}
STOREFRONT_COUNT = 84
INTERIOR_COUNT = 18
WALKUP_COUNT = 3
GROUND_FLOOR_H = 4.5
UPPER_FLOOR_H = 3.2
SEED = 1782

LOOT_TARGET = 110
ZOMBIE_SPAWNERS = 12

# --- materials (slot → blockout RGBA; UE binds material instances by slot name) -------
MATERIALS = {
    "M_Concrete": (0.42, 0.42, 0.40, 1), "M_Tile_White": (0.85, 0.84, 0.78, 1),
    "M_Tile_Trim": (0.30, 0.45, 0.35, 1), "M_Terrazzo": (0.55, 0.52, 0.47, 1),
    "M_Platform_Edge": (0.85, 0.65, 0.05, 1), "M_Trackbed": (0.12, 0.10, 0.09, 1),
    "M_Steel_Painted": (0.18, 0.25, 0.22, 1), "M_Asphalt": (0.10, 0.10, 0.11, 1),
    "M_Sidewalk": (0.55, 0.54, 0.51, 1), "M_Brick": (0.45, 0.22, 0.15, 1),
    "M_Glass_Dark": (0.05, 0.07, 0.08, 1), "M_Wood": (0.40, 0.27, 0.16, 1),
    "M_Sandbag": (0.52, 0.47, 0.35, 1), "M_Train_Body": (0.62, 0.64, 0.66, 1),
    "M_Globe_Green": (0.10, 0.80, 0.25, 1), "M_Globe_Red": (0.85, 0.10, 0.08, 1),
    "M_Light_Emissive": (1.00, 0.95, 0.85, 1),
}
