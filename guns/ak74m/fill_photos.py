"""fill_photos.py - PRIVATE helper for projpaint.py: copies of both profile photos with the gun's edge colours grown over the
white background (no background bleed on grazing faces). Writes work/proj_{L,R}_filled.png."""
import os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ak_ref as R

for ph, side in ((R.PHOTO, "L"), (R.PHOTO_R, "R")):
    fg, a = R.masks(ph)
    _, (iy, ix) = ndimage.distance_transform_edt(~ndimage.binary_erosion(fg, iterations=2), return_indices=True)
    Image.fromarray(a[iy, ix].astype(np.uint8)).save(os.path.join(HERE, "work", f"proj_{side}_filled.png"))
