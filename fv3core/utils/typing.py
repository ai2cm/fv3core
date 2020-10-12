from typing import Tuple

from gtscript import Field

import numpy as np


Int3 = Tuple[int, int, int]
"""Common type: tuple of three ints."""


class _FieldDescriptor:
    def __getitem__(self, dtype_and_axes):
        return gtscript.Field[dtype_and_axes]


Field = _FieldDescriptor()
"""A gtscript field."""

# Typing shortcuts that should be used instead
FField = Field[np.float_]
BField = Field[bool]
IField = Field[int]
