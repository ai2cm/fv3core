import fv3core.stencils.dyn_core as dyn_core
from fv3core.testing import TranslateFortranData2Py


class TranslatePGradC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "delpc": {},
            "pkc": grid.default_buffer_k_dict(),
            "gz": grid.default_buffer_k_dict(),
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}

    def compute_from_storage(self, inputs):
        dyn_core.p_grad_c_stencil(
            rdxc=self.grid.rdxc,
            rdyc=self.grid.rdyc,
            **inputs,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        return inputs
