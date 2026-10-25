"""scar_spec.py - SCAR 17S (NRCH, 16.25 in, FDE) dimensions in the B frame (mm; x gun-LEFT, y muzzle, z up;
bore axis = z 0, bolt face = y 0). The game frame G puts the origin at the top of the pistol grip: G = B - ORIGIN_B.

Variant lock: FN SCAR 17S NRCH, 7.62x51, 16.25 in cold-hammer-forged barrel (1:12 RH), MIL-STD-1913 rails at
3/6/9/12 o'clock, side-folding telescoping polymer stock with cheek riser, A2-style grip, enlarged trigger guard,
ambidextrous safety + mag release, non-reciprocating charging handle (left), adjustable 2-position gas regulator,
folding front + rear sights, 20-round steel magazine. Published: 8.0 lb, 38.5 in OAL extended / 28.5 in folded.
No reference photos could be fetched from this environment (network policy), so feature shapes are from the
published specs and the well-known layout; S2 silhouette IoU / S5 diffs are OPEN until the user supplies photos
(see STATUS.md). No markings, logos or serials are modelled."""

IN = 25.4
BARREL_L = 16.25 * IN                  # 412.75 from the bolt face
MUZZLE_Y = 456.0                       # flash hider face
OAL_EXT = 38.5 * IN                    # 977.9 -> buttpad rear at MUZZLE_Y - OAL_EXT
BUTT_Y = MUZZLE_Y - OAL_EXT            # -521.9
OAL_FOLD = 28.5 * IN

# receiver
UP_Y0, UP_Y1 = -178.0, 292.0           # upper: rear face (hinge block behind it) .. forend front
UP_HW, FE_HW = 26.0, 27.0              # half widths: receiver, forend
UP_TOP, UP_BOT, FE_BOT = 24.0, -21.0, -35.0
RAIL_TOP = 31.9                        # 1913 rail top above the bore
RAIL_PITCH, RAIL_SLOT_W, RAIL_SLOT_D = 10.01, 5.23, 2.9
SIDE_RAIL_Y = (128.0, 286.0)
SIDE_RAIL_Z = -4.0
PORT = dict(y=(-45.0, 50.0), z=(-5.0, 15.0))          # ejection port, right side (x < 0)
CH_SLOT = dict(y=(-55.0, 226.0), z=(5.0, 12.0))      # charging handle slot, left side (x > 0)
CH_FWD_Y = 212.0                                     # handle centre, forward (rest)
CH_TRAVEL = 255.0
BOLT_TRAVEL = 118.0

# lower + grip
LO_Y0, LO_Y1, LO_BOT = -178.0, 28.0, -53.0
MAG_Y = (-86.5, -7.5)                  # magazine body rear/front at the top
MAG_HW = 13.0
MAG_TOP, MAG_BOT = -27.0, -176.0
GRIP_TOP = (-166.0, -42.0)             # (y, z) grip axis top
GRIP_BOT = (-197.0, -150.0)
TRIGGER_Y = -110.0

# stock (extended)
HINGE = (-24.0, -196.0)                # (x, y) of the vertical hinge axis, folds to the right
STOCK_Y0 = -200.0

ORIGIN_B = (0.0, GRIP_TOP[0], GRIP_TOP[1])   # game origin = grip top


# ---- v2: measured on the profile master (ref.py, 0.5605 mm/px, bolt face = front of the ejection port) ----
MUZZLE_Y = 463.0                       # brake face
RAIL_TOP = 53.0
MAG_Y = (-98.1, -21.9)                 # magazine body rear / front
MAG_TOP = -21.0
BUTT_Y = -490.5
HINGE = (-23.0, -257.3)
SIDE_RAIL_Z = -3.6                     # side M-LOK row centre
FE_BOT = -26.1                         # handguard bottom
GRIP_TOP = (-194.5, -39.0)
ORIGIN_B = (0.0, -194.5, -39.0)
FRONT_SIGHT_TOP = (279.1, 84.4)
REAR_APERTURE = (-214.7, 82.1)
VGRIP = (195.1, -27.7)
HINGE = (-23.0 * 0.75, -257.3)        # width-scaled (silfit)
