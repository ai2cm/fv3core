from .translate import TranslateFortranData2Py
import fv3.stencils.map_ppm_2d as Map1_PPM_2d
import numpy as np


class TranslateMap1_PPM_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Map1_PPM_2d.compute
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "var_in"},
            "pe1": {"istart": 3, "iend": grid.ie - 2},
            "pe2": {"istart": 3, "iend": grid.ie - 2},
            "qs": {"serialname": "ws_1d"},
        }
        self.in_vars["parameters"] = ["j_2d", "i1", "i2", "mode", "kord"]
        self.out_vars = {
            "var_inout": {},
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
        inputs["i1"] += 2
        inputs["i2"] += 2
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"var_inout": var_inout})


class TranslateMap1_PPM_2d_3(TranslateMap1_PPM_2d):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["pe1"]["serialname"] = "pe1_2"
        self.in_vars["data_vars"]["pe2"]["serialname"] = "pe2_2"
        self.in_vars["data_vars"]["q1"]["serialname"] = "var_inout_3"
        self.out_vars = {
            "var_inout": {
                "serialname": "var_inout_3",
                "istart": 0,
                "iend": grid.ied + 1,
            },
        }
