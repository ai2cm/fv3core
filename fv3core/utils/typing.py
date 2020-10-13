from typing import Tuple

import gt4py.gtscript as gtscript
import numpy as np


__all__ = [
    "Int3",
    "Field",
    "FloatField",
    "BoolField",
    "IntField",
    "IJK",
    "IJ",
    "IK",
    "JK",
    "I",
    "J",
    "K",
]

Int3 = Tuple[int, int, int]
"""Common type: tuple of three ints."""

Field = gtscript.Field
"""A gt4py field"""


class _FieldDescriptorMaker:
    """Shortcut for float fields"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, axes):
        return gtscript.Field[self.dtype, axes]


# These could be used, but they seem just as long not using
FloatField = _FieldDescriptorMaker(float)
IntField = _FieldDescriptorMaker(int)
BoolField = _FieldDescriptorMaker(bool)

# Axes
IJK = gtscript.IJK
IJ = gtscript.IJ
IK = gtscript.IK
JK = gtscript.JK
I = gtscript.I
J = gtscript.J
K = gtscript.K

# Usage example:
# from fv3core.utils.typing import Field, IJK, IJ
# def stencil(in_field: Field[float, IJ], out_field: Field[float, IJK]):
