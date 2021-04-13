import numpy as np

import fv3core.stencils.map_single as map_single
from fv3core.testing import TranslateFortranData2Py, TranslateGrid


def pad_field_in_j(field, nj):
    outfield = np.tile(field[:, 0, :], [nj, 1, 1]).transpose(1, 0, 2)
    np.testing.assert_array_equal(outfield[:, 0, :], field[:, 0, :])
    return outfield


class TranslateSingleJ(TranslateFortranData2Py):
    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None
            info["dummy_axes"] = dummy_axes
        super().make_storage_data_input_vars(inputs, storage_vars)


class TranslateMap1_PPM_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = map_single.compute
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "var_in"},
            "pe1": {"istart": 3, "iend": grid.ie - 2, "axis": 1},
            "pe2": {"istart": 3, "iend": grid.ie - 2, "axis": 1},
            "qs": {"serialname": "ws_1d", "kstart": grid.is_, "axis": 0},
        }
        self.in_vars["parameters"] = ["j_2d", "i1", "i2", "mode", "kord"]
        self.out_vars = {"var_inout": {}}
        self.max_error = 5e-13
        self.write_vars = ["qs"]
        self.nj = grid.npy

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        if "qs" in inputs:
            qs_3d = pad_field_in_j(inputs["qs"].data, self.nj)
            inputs["qs"] = self.make_storage_data(qs_3d)
        inputs["i1"] = self.grid.global_to_local_x(
            inputs["i1"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i2"] = self.grid.global_to_local_x(
            inputs["i2"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["qmin"] = 0.0
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["j1"] = inputs["j_2d"]
        inputs["j2"] = inputs["j_2d"]
        del inputs["j_2d"]
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"var_inout": var_inout})


class TranslateMap1_PPM_2d_3(TranslateMap1_PPM_2d):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["pe1"]["serialname"] = "pe1_2"
        self.in_vars["data_vars"]["pe2"]["serialname"] = "pe2_2"
        self.in_vars["data_vars"]["q1"]["serialname"] = "var_in_3"
        self.out_vars = {
            "var_inout": {
                "serialname": "var_inout_3",
                "istart": 0,
                "iend": grid.ied + 1,
            }
        }


class TranslateMap1_PPM_2d_2(TranslateMap1_PPM_2d):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["pe1"]["serialname"] = "pe1_2"
        self.in_vars["data_vars"]["pe2"]["serialname"] = "pe2_2"
        self.in_vars["data_vars"]["q1"]["serialname"] = "var_in_2"
        self.out_vars = {
            "var_inout": {
                "serialname": "var_inout_2",
                "jstart": 0,
                "jend": grid.jed + 1,
            }
        }
        self.max_error = 2e-14
