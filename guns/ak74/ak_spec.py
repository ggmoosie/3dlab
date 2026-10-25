"""ak_spec.py - AK-74M dimensions in the B frame (mm; x gun-LEFT, y toward the muzzle, z up; bore axis z 0; y 0 = receiver
rear face), measured on the right-profile master (ak_ref.py, 0.9624 mm/px) and the close-ups. Game frame G = B - ORIGIN_B
(origin at the grip top, like the SCAR). Variant: AK-74M, 5.45x39, 415 mm barrel, black polymer furniture, side-folding
(to the left) polymer stock, 30-round polymer magazine, left-side optics rail. No factory marks: generic serial only."""
import ak_ref as R

S = R.S_MM
MUZZLE_Y = R.MUZZLE_Y                       # netto ~693.8
BUTT_Y = R.BUTT_Y                           # ~ -249.2
OAL = 943.0
OAL_FOLDED = 705.0

# cross-section half widths (x), first estimates from the 3/4 shots and the front view; refined by silfit
HW_REC = 15.6          # stamped receiver outside (1.0 mm sheet)
HW_DC = 16.3           # dust cover skirt, outside the receiver wall
HW_RSB = 15.6
HW_UHG = 15.5          # upper handguard (gas tube cover)
HW_LHG = 21.0          # lower handguard at its widest
HW_GRIP = 14.0
HW_MAG = 13.6
HW_STOCK_BUTT = 20.5
HW_STOCK_WRIST = 16.0

BARREL_R = 6.55        # between gas block and FSB (13.3 px)
GAS_TUBE_R = 8.85
ROD_R = 1.75           # cleaning rod (297.2..300.8)

# landmarks (y, z) from photo px
def P(x, row):
    return R.bp(x, row)

GRIP_TOP = P(500.0, 316.5)
ORIGIN_B = (0.0, GRIP_TOP[0], GRIP_TOP[1])
MAG_TOP = P(641.0, 312.0)                  # magazine top centre (in the magwell)
MAG_PIVOT = P(684.0, 312.0)                # front lug: the magazine rocks in about the front catch lug
FRONT_SIGHT_TOP = P(1084.0, 233.0)
REAR_NOTCH = P(699.0, 231.5)
HINGE = (16.6, -3.0)                       # (x, y) of the vertical stock hinge axis (left side, behind the receiver)
DC_PIVOT = P(462.0, 262.0)                 # dust cover lifts about its rear (guide-rod button)
SELECTOR_AXIS = P(517.0, 288.0)
CH_POS = P(694.0, 265.0)                   # charging handle (on the carrier), right side
CARRIER_TRAVEL = 0.0                       # set below from the receiver length
TRIGGER_PIVOT = P(546.0, 306.0)
EJECTION = P(650.0, 267.0)
