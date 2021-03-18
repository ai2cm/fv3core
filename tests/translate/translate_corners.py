import numpy as np

from fv3core.testing import TranslateFortranData2Py
from fv3core.utils import corners


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
