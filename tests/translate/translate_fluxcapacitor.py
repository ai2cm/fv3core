import fv3core.stencils.d_sw as d_sw
from fv3core.testing import TranslateFortranData2Py


class TranslateFluxCapacitor(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "xflux": grid.x3d_compute_dict(),
            "yflux": grid.y3d_compute_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
        }
        self.out_vars = {}
        for outvar in ["cx", "cy", "xflux", "yflux"]:
            self.out_vars[outvar] = self.in_vars["data_vars"][outvar]

    def compute_from_storage(self, inputs):
        d_sw.flux_capacitor(
            **inputs,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )
        return inputs
