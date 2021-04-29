from fv3core.testing.translate import MockNamelist
from fv3core.testing import TranslateFortranData2Py


from fv3core.stencils.dyn_core import AcousticDynamics
import fv3core._config as spec


class TranslatePressureAdjustedTemperature_NonHydrostatic(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

        namelist = MockNamelist()
        acoustic_dynamics = AcousticDynamics(None, namelist, None, None, None)
        self.compute_func = acoustic_dynamics._compute_pkz_tempadjust
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
