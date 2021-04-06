import numpy as np

import fv3core.stencils.remap_profile as profile
import fv3core.utils.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py

def pad_data_in_j(ik_data, nj):
    full_data = np.tile(ik_data, [nj,1,1]).transpose(1,0,2)
    np.testing.assert_array_equal(full_data[:,0,:], ik_data, err_msg="Padded field is wrong")
    return full_data

class TranslateCS_Profile_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = profile.compute
        self.in_vars["data_vars"] = {
            "a4_1": {"serialname": "q4_1"},
            "a4_2": {"serialname": "q4_2"},
            "a4_3": {"serialname": "q4_3"},
            "a4_4": {"serialname": "q4_4"},
            "delp": {"serialname": "dp1_2d"},
        }
        self.in_vars["parameters"] = ["km", "i1", "i2", "iv", "kord"]
        self.out_vars = {
            "a4_1": {"serialname": "q4_1", "istart": 0, "iend": grid.ie - 2, "jstart": grid.js, "jend": grid.js},
            "a4_2": {"serialname": "q4_2", "istart": 0, "iend": grid.ie - 2, "jstart": grid.js, "jend": grid.js},
            "a4_3": {"serialname": "q4_3", "istart": 0, "iend": grid.ie - 2, "jstart": grid.js, "jend": grid.js},
            "a4_4": {"serialname": "q4_4", "istart": 0, "iend": grid.ie - 2, "jstart": grid.js, "jend": grid.js},
        }
        self.ignore_near_zero_errors = {"q4_4": True}
        self.write_vars = ["qs"]
        self.nj = grid.npy

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )
            if len(np.squeeze(inputs[serialname]).shape) == 2:
                inputs[serialname] = pad_data_in_j(inputs[serialname], self.nj)
            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                axis=len(inputs[serialname].shape) - 1,
                read_only=d not in self.write_vars,
            )
            if d != serialname:
                del inputs[serialname]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["i1"] = self.grid.global_to_local_x(inputs["i1"] - 1)
        inputs["i2"] = self.grid.global_to_local_x(inputs["i2"] - 1)
        if "qs" not in inputs:
            inputs["qs"] = utils.make_storage_from_shape(self.maxshape)
        else:
            qs_field = utils.make_storage_from_shape(
                inputs["delp"].shape, origin=(0, 0, 0)
            )
            qs_field[inputs["i1"] : inputs["i2"] + 1, inputs["jslice"], -1] = inputs[
                "qs"
            ][inputs["i1"] : inputs["i2"] + 1, inputs["jslice"], 0]
            inputs["qs"] = qs_field
        q4_1, q4_2, q4_3, q4_4 = self.compute_func(**inputs)
        return self.slice_output(
            inputs, {"q4_1": q4_1, "q4_2": q4_2, "q4_3": q4_3, "q4_4": q4_4}
        )


class TranslateCS_Profile_2d_2(TranslateCS_Profile_2d):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = profile.compute
        self.in_vars["data_vars"] = {
            "qs": {"serialname": "qs_column_2", "kstart": 0, "kend": grid.npz},
            "a4_1": {"serialname": "q4_1_2"},
            "a4_2": {"serialname": "q4_2_2"},
            "a4_3": {"serialname": "q4_3_2"},
            "a4_4": {"serialname": "q4_4_2"},
            "delp": {"serialname": "dp1_2d_2"},
        }
        self.in_vars["parameters"] = ["km", "i1", "i2", "iv", "kord"]
        self.out_vars = {
            "a4_1": {"serialname": "q4_1_2", "istart": 0, "iend": grid.ie - 3},
            "a4_2": {"serialname": "q4_2_2", "istart": 0, "iend": grid.ie - 3},
            "a4_3": {"serialname": "q4_3_2", "istart": 0, "iend": grid.ie - 3},
            "a4_4": {"serialname": "q4_4_2", "istart": 0, "iend": grid.ie - 3},
        }
        self.ignore_near_zero_errors = {"q4_4_2": True}
