from fv3core.testing import TranslateFortranData2Py
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.testing.translate import MockNamelist


class TranslatePE_Halo(TranslateFortranData2Py):
    def __init__(self, grid):

        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "delp": {},
        }
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {"pe": self.in_vars["data_vars"]["pe"]}

        namelist = MockNamelist()
        acoustic_dynamics = AcousticDynamics(None, namelist, None, None, None)
        self.compute_func = acoustic_dynamics._edge_pe_stencil
