from .translate import TranslateFortranData2Py
import fv3.stencils.cs_profile as CS_Profile
import numpy as np


class TranslateCS_Profile_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = CS_Profile.compute
        self.in_vars["data_vars"] = {
            "qs": {"serialname": "qs_column"},
            "a4_1": {"serialname": "q4_1"},
            "a4_2": {"serialname": "q4_2"},
            "a4_3": {"serialname": "q4_3"},
            "a4_4": {"serialname": "q4_4"},
            "delp": {"serialname": "dp1_2d"},
            "set_gam":{}, 
            "set_q":{},
            "set_a4":{}, 
            "b_q":{}, 
            "b_gam":{},
            "b_a4":{}, 
            "b_extm":{}, 
            "b_ext5":{}, 
            "b_ext6":{},
            "cs1_extm":{},
            "cs1_a4_1":{},
            "cs1_a4_2":{},
            "cs1_a4_3":{},
            "cs1_a4_4":{},
            "cs1b_a4_1":{},
            "cs1b_a4_2":{},
            "cs1b_a4_3":{},
            "cs1b_a4_4":{},
            "cs2_extm":{},
            "cs2_a4_1":{},
            "cs2_a4_2":{},
            "cs2_a4_3":{},
            "cs2_a4_4":{},
            "cs2b_a4_1":{},
            "cs2b_a4_2":{},
            "cs2b_a4_3":{},
            "cs2b_a4_4":{},
            "huy_a4_1":{},
            "huy_a4_2":{},
            "huy_a4_3":{},
            "huy_a4_4":{},
        }
        self.in_vars["parameters"] = ["km", "i1", "i2", "iv", "kord"]
        self.out_vars = {
            "a4_1": {"serialname": "q4_1", "istart": 0,"iend": grid.ie-2},
            "a4_2": {"serialname": "q4_2", "istart": 0,"iend": grid.ie-2},
            "a4_3": {"serialname": "q4_3", "istart": 0,"iend": grid.ie-2},
            "a4_4": {"serialname": "q4_4", "istart": 0,"iend": grid.ie-2},
        }

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
            print(serialname)

            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None

            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
            )
            if d != serialname:
                del inputs[serialname]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        q4_1, q4_2, q4_3, q4_4 = self.compute_func(**inputs)
        return self.slice_output(
            inputs, {"q4_1": q4_1, "q4_2": q4_2, "q4_3": q4_3, "q4_4": q4_4}
        )
