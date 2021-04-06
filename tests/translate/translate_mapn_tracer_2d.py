import fv3core._config as spec
import numpy as np
import fv3core.utils.gt4py_utils as utils
import fv3core.stencils.mapn_tracer as MapN_Tracer
from fv3core.testing import TranslateFortranData2Py, TranslateGrid
from tests.translate.translate_remap_profile_2d import pad_data_in_j

class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = MapN_Tracer.compute
        self.in_vars["data_vars"] = {
            "pe1": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "pe2": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "dp2": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "tracers": {"serialname": "qtracers"},
        }
        self.in_vars["parameters"] = ["j_2d", "nq", "q_min"]
        self.out_vars = {"tracers": {"serialname": "qtracers"}}

        self.is_ = grid.is_
        self.ie = grid.ie
        self.max_error = 3.5e-11
        self.near_zero = 7e-17
        self.ignore_near_zero_errors["qtracers"] = True
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
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )
            
            if len(np.squeeze(inputs[serialname]).shape) == 2:
                inputs[serialname] = pad_data_in_j(inputs[serialname], self.nj)

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
        inputs["kord"] = abs(spec.namelist.kord_tr)
        self.compute_func(**inputs)
        return self.slice_output(inputs)
