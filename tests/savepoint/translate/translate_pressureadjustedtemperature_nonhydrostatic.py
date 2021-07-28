import fv3core._config as spec
from fv3core.stencils.dyn_core import (
    _initialize_temp_adjust_stencil,
    get_nk_heat_dissipation,
)
from fv3core.testing import TranslateFortranData2Py


class TranslatePressureAdjustedTemperature_NonHydrostatic(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        n_adj = get_nk_heat_dissipation(
            config=spec.namelist.acoustic_dynamics.d_sw,
            npz=grid.grid_indexing.domain[2],
        )
        self.compute_func = _initialize_temp_adjust_stencil(grid.grid_indexing, n_adj)
        self.in_vars["data_vars"] = {
            "cappa": {},
            "delp": {},
            "delz": {},
            "pt": {},
            "heat_source": {"serialname": "heat_source_dyn"},
            "pkz": grid.compute_dict(),
        }
        self.in_vars["parameters"] = ["bdt"]
        self.out_vars = {"pt": {}, "pkz": grid.compute_dict()}

    def compute_from_storage(self, inputs):
        inputs["delt_time_factor"] = abs(inputs["bdt"] * spec.namelist.delt_max)
        del inputs["bdt"]
        self.compute_func(**inputs)
        return inputs
