from typing import Tuple

import gt4py.gtscript as gtscript
import numpy as np


Field = gtscript.Field
"""A gt4py field"""

# Axes
IJK = gtscript.IJK
IJ = gtscript.IJ
IK = gtscript.IK
JK = gtscript.JK
I = gtscript.I
J = gtscript.J
K = gtscript.K

# Union of valid data types (from gt4py.gtscript)
DTypes = Union[bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64]

# Other common types
Int3 = Tuple[int, int, int]
"""Common type: tuple of three ints."""
