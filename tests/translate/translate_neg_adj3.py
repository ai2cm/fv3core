import pytest

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from fv3core.testing import TranslateFortranData2Py


class TranslateNeg_Adj3(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "pt": {},
            "delp": {},
            "delz": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            # "pt": {},
        }
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator
        state = self.state_from_inputs(inputs)
        compute_fn = AdjustNegativeTracerMixingRatio(
            self.grid, spec.namelist, inputs["qvapor"], inputs["qgraupel"]
        )
        compute_fn(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            inputs["qcld"],
            inputs["pt"],
            inputs["delp"],
            inputs["delz"],
            inputs["peln"],
        )
        outputs = self.outputs_from_state(state)
        return outputs

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )
