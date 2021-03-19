import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

from fv3core.decorators import gtstencil
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils import corners
from fv3core.utils.typing import FloatField


@gtstencil
def fill_4corners_x_2cells(q: FloatField):
    with computation(PARALLEL), interval(...):
        q = corners.fill_corners_2cells_mult_x(q, q, 1.0, 1.0, 1.0, 1.0)


@gtstencil
def fill_4corners_y_2cells(q: FloatField):
    with computation(PARALLEL), interval(...):
        q = corners.fill_corners_2cells_mult_y(q, q, 1.0, 1.0, 1.0, 1.0)


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute_from_storage(self, inputs):
        if inputs["dir"] == 1:
            fill_4corners_x_2cells(
                inputs["q4c"],
                origin=self.grid.full_origin(),
                domain=self.grid.domain_shape_full(),
            )
        elif inputs["dir"] == 2:
            fill_4corners_y_2cells(
                inputs["q4c"],
                origin=self.grid.full_origin(),
                domain=self.grid.domain_shape_full(),
            )
        else:
            raise ValueError("Direction not recognized. Specify either x or y")
        return inputs


class TranslateFillCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"divg_d": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}

    def compute(self, inputs):
        if inputs["dir"] == 1:
            direction = "x"
        elif inputs["dir"] == 2:
            direction = "y"
        else:
            raise ValueError("Invalid input")
        nord_column = inputs["nord_col"][0, 0, :]
        self.make_storage_data_input_vars(inputs)
        for nord in np.unique(nord_column):
            if nord != 0:
                ki = [i for i in range(self.grid.npz) if nord_column[i] == nord]
                corners.fill_corners_2d(
                    inputs["divg_d"],
                    self.grid,
                    "B",
                    direction,
                    kstart=ki[0],
                    nk=len(ki),
                )
        return self.slice_output(inputs)


class TranslateCopyCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q": {}}

    def compute_from_storage(self, inputs):
        if inputs["dir"] == 1:
            corners.copy_corners_x_stencil(
                inputs["q"],
                origin=self.grid.full_origin(),
                domain=self.grid.domain_shape_full(),
            )
        elif inputs["dir"] == 2:
            corners.copy_corners_y_stencil(
                inputs["q"],
                origin=self.grid.full_origin(),
                domain=self.grid.domain_shape_full(),
            )
        else:
            raise ValueError("Invalid input")
        return inputs


class TranslateFillCornersVector(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"vc": {}, "uc": {}, "nord_col": {}}
        self.out_vars = {"vc": grid.y3d_domain_dict(), "uc": grid.x3d_domain_dict()}

    def compute(self, inputs):
        nord_column = inputs["nord_col"][0, 0, :]
        vector = True
        self.make_storage_data_input_vars(inputs)
        for nord in np.unique(nord_column):
            if nord != 0:
                ki = [i for i in range(self.grid.npz) if nord_column[i] == nord]
                corners.fill_corners_dgrid(
                    inputs["vc"],
                    inputs["uc"],
                    self.grid,
                    vector,
                    kstart=ki[0],
                    nk=len(ki),
                )
        return self.slice_output(inputs)
