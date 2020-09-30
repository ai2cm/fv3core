from typing import Tuple

from gtscript import Field


Int3 = Tuple[int, int, int]
"""Common type: tuple of three ints."""


class _FieldDescriptor:
    def __getitem__(self, dtype_and_axes):
        return gtscript.Field[dtype_and_axes]


Field = _FieldDescriptor()
"""A gtscript field."""
