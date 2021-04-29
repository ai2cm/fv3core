from fv3core.testing import TranslateFortranData2Py
from fv3core.stencils.dyn_core import AcousticDynamics


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

        self.compute_func = AcousticDynamics.initialize_edge_pe_stencil(grid)
