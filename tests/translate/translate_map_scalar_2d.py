import fv3core._config as spec
from fv3core.stencils.map_single import MapSingleFactory
from fv3core.testing import TranslateFortranData2Py, TranslateGrid


class TranslateMapScalar_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = MapSingleFactory()
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
        self.out_vars = {"pt": {}}  # "jstart": grid.js, "jend": grid.js
        self.is_ = grid.is_
        self.ie = grid.ie
        self.write_vars = ["qs"]
        self.nj = grid.npy
        self.nk = grid.npz

    def compute(self, inputs):
        self.setup(inputs)
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i1"] = self.is_
        inputs["i2"] = self.ie
        inputs["j1"] = inputs["j_2d"]
        inputs["j2"] = inputs["j_2d"]
        del inputs["j_2d"]
        inputs["kord"] = abs(spec.namelist.kord_tm)
        inputs["qmin"] = 184.0
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"pt": var_inout})
