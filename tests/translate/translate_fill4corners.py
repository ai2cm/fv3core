import fv3core.utils.gt4py_utils as utils
from fv3core.utils import corners

from .translate import TranslateFortranData2Py


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):
        inputs["q4c"] = utils.make_storage_data(inputs["q4c"], inputs["q4c"].shape)
        corners.fill_4corners(
            inputs["q4c"], "x" if inputs["dir"] == 1 else "y", self.grid
        )
        return {"q4c": inputs["q4c"]}
