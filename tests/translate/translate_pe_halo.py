import fv3core.stencils.pe_halo as pe_halo
from fv3core.decorators import StencilWrapper
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.grid import axis_offsets
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

        class MockNamelist:
            def __init__(self) -> None:
                self.hydrostatic = True
                self.d_ext = 0
                self.beta = 0
                self.use_logp = False
                self.convert_ke = True

        namelist = MockNamelist()
        acoustic_dynamics = AcousticDynamics(None, namelist, None, None, None)
        self.compute_func = acoustic_dynamics._edge_pe_stencil
