from typing import Tuple

import numpy as np


class SelectiveValidation:
    """
    Class used to validate slices of certain outputs instead of the whole
    output. Makes sure these non-validated outputs are not actually used by
    setting them to NaN during validation tests, which will cause
    test failures otherwise.
    """

    TEST_MODE = False

    def __init__(self, origin: Tuple[int, ...], domain: Tuple[int, ...]):
        self._validation_slice = tuple(
            slice(start, start + n) for start, n in zip(origin, domain)
        )
        self.origin = origin
        self.domain = domain

    @property
    def translate_dict(self):
        return {
            "i_start": self.origin[0],
            "j_start": self.origin[1],
            "k_start": self.origin[2],
            "i_end": self.origin[0] + self.domain[0],
            "j_end": self.origin[1] + self.domain[1],
            "k_end": self.origin[2] + self.domain[2],
        }

    @property
    def validation_slice(self) -> Tuple[slice, ...]:
        return self._validation_slice

    def set_nans_if_test_mode(self, array):
        if SelectiveValidation.TEST_MODE:
            validation_data = np.copy(array[self.validation_slice])
            array[:] = np.nan
            array[self.validation_slice] = validation_data
