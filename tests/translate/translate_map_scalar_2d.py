import fv3core._config as spec
import numpy as np
import fv3core.stencils.map_single as Map_Single
import fv3core.utils.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py, TranslateGrid
from tests.translate.translate_remap_profile_2d import pad_data_in_j

def pad_1d_data(in_data, n1, n2, axis=0):
    if axis == 0: #n2 = k, n1 = j
        out_data = np.tile(in_data, [n2, n1, 1]).transpose(2, 1, 0)
        np.testing.assert_array_equal(out_data[:,0,0], in_data, err_msg="Padded field is wrong")
    elif axis == 1: #n2 = k, n1 = i
        out_data = np.tile(in_data, [n2, n1, 1]).transpose(2, 0, 1)
        np.testing.assert_array_equal(out_data[0,:,0], in_data, err_msg="Padded field is wrong")
    elif axis == 2: #n2 = j, n1 = i
        out_data = np.tile(in_data, [n2, n1, 1]).transpose(1, 0, 2)
        np.testing.assert_array_equal(out_data[0,0,:], in_data, err_msg="Padded field is wrong")
    else:
        raise ValueError("Axis must be 0, 1, or 2")
    return out_data

class TranslateMapScalar_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Map_Single.compute
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "pt"},
            "pe1": {
                "serialname": "peln",
                "istart": grid.is_,
                "iend": grid.ie - 2,
                "kaxis": 1,
                "axis": 1,
            },
            "pe2": {
                "istart": grid.is_,
                "iend": grid.ie - 2,
                "serialname": "pn2",
                "axis": 1,
            },
            "qs": {"serialname": "gz1d", "kstart": 0, "axis": 0},
        }
        self.in_vars["parameters"] = ["j_2d", "mode"]
        self.out_vars = {"pt": {}} # "jstart": grid.js, "jend": grid.js
        self.is_ = grid.is_
        self.ie = grid.ie
        self.write_vars = ["qs"]
        self.nj = grid.npy
        self.nk = grid.npz

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs)
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )
            
            if len(np.squeeze(inputs[serialname]).shape) == 2:
                inputs[serialname] = pad_data_in_j(inputs[serialname], self.nj)
            if serialname == "gz1d":
                inputs[serialname] = pad_1d_data(inputs[serialname], self.nj, self.nk+1)



            names_4d = None
            if len(inputs[serialname].shape) == 4:
                names_4d = info.get("names_4d", utils.tracer_variables)

            dummy_axes = info.get("dummy_axes", None)
            axis = info.get("axis", 2)
            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
                axis=axis,
                names_4d=names_4d,
                read_only=d not in self.write_vars,
                full_shape="full_shape" in storage_vars[d],
            )
            
            if d != serialname:
                del inputs[serialname]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i1"] = self.is_
        inputs["i2"] = self.ie
        inputs["kord"] = abs(spec.namelist.kord_tm)
        inputs["qmin"] = 184.0
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"pt": var_inout})
