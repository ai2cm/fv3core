from fv3core.utils.corners import fill_4corners
from gt4py import gtscript

from .translate import TranslateFortranData2Py
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils import corners


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):

        def definition(field: utils.sd):
            from __externals__ import func
            with computation(PARALLEL), interval(...):
                field = func(field)

        kwargs = {
            "origin": self.grid.compute_origin(add=(-3, -3, 0)),
            "domain": self.grid.domain_shape_compute_buffer_2d(add=(6,6,0))
        }
        if inputs["dir"] == 1:
            stencil = gtstencil(definition=definition, externals={"func": corners.fill_4corners_x_func})
            stencil(inputs["q4c"], **kwargs)
        elif inputs["dir"] == 2:
            stencil = gtstencil(definition=definition, externals={"func": corners.fill_4corners_y_func})
            stencil(inputs["q4c"], **kwargs)

        return {"q4c": inputs["q4c"]}
