from .translate import TranslateFortranData2Py
import fv3.stencils.map_scalar as Map_Scalar
import fv3.stencils.map_single as Map_Single
import numpy as np
import fv3._config as spec


class TranslateMapScalar_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Map_Single.compute
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "pt"},
            "pe1": {"serialname": "peln", "istart": grid.is_, "iend": grid.ie - 2, "kaxis": 1},
            "pe2": {"istart": grid.is_, "iend": grid.ie - 2, "serialname": "pn2"},
            "qs": {"serialname": "gz1d", 'kstart': grid.is_, 'axis': 0},
        }
        self.in_vars["parameters"] = ["j_2d", "mode"]
        self.out_vars = {
            "pt": {},
        }
        self.is_ = grid.is_
        self.ie = grid.ie

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

            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None
            axis = info.get("axis", 2)
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
        inputs["j_2d"] += 2
        inputs["i1"] = self.is_
        inputs["i2"] = self.ie
        inputs["kord"] = abs(spec.namelist["kord_tm"])
        inputs["qmin"] = 184.0
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"pt": var_inout})
