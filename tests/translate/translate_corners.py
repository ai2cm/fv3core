import numpy as np

import fv3core.utils.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils import corners
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField

class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        corners.fill_corners_cells(inputs["q4c"], "x" if inputs["dir"] == 1 else "y")
        return self.slice_output(inputs, {"q4c": inputs["q4c"]})

@gtstencil
def fill_corners_2d_bgrid_x_stencil(q: FloatField):
    with computation(PARALLEL), interval(3, None):
        q = corners.fill_corners_2d_bgrid_x(q, q)
@gtstencil
def fill_corners_2d_bgrid_y_stencil(q: FloatField):
    with computation(PARALLEL), interval(3, None):
        q = corners.fill_corners_2d_bgrid_y(q, q)

class TranslateFillCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"divg_d": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}

    def compute_from_storage(self, inputs):
        if  inputs["dir"] == 1:
            fill_corners_2d_bgrid_x_stencil(
                inputs["divg_d"], origin=self.grid.full_origin(), domain=self.grid.domain_shape_full(add=(1, 1, 0)))
        elif  inputs["dir"] == 2:
             fill_corners_2d_bgrid_y_stencil(
                inputs["divg_d"], origin=self.grid.full_origin(), domain=self.grid.domain_shape_full(add=(1, 1, 0)
            ))
        else:
            raise ValueError("Invalid input")
        return inputs


class TranslateCopyCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q": {}}

    def compute(self, inputs):
        if inputs["dir"] == 1:
            direction = "x"
        elif inputs["dir"] == 2:
            direction = "y"
        else:
            raise ValueError("Invalid input")
        corners.copy_corners(inputs["q"], direction, self.grid)
        return {"q": inputs["q"]}

@gtstencil
def fill_corners_dgrid_stencil(u: FloatField, v: FloatField, mysign: float):
    with computation(PARALLEL), interval(3, None):
        u, v, = corners.fill_corners_dgrid_fn(u, v, mysign)

class TranslateFillCornersVector(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"vc": {}, "uc": {}}
        self.out_vars = {"vc": grid.y3d_domain_dict(), "uc": grid.x3d_domain_dict()}

    def compute_from_storage(self, inputs):
        mysign = -1.0
        fill_corners_dgrid_stencil(inputs["vc"], inputs["uc"], mysign, origin=self.grid.full_origin(), domain=self.grid.domain_shape_full(add=(1, 1, 0)))
        return inputs
