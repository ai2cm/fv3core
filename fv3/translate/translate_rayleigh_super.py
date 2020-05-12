from .translate import TranslateFortranData2Py
import fv3.stencils.rayleigh_super as super_ray


class TranslateRayleigh_Super(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = super_ray.compute
        self.in_vars["data_vars"] = {
            "phis": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "ua": {},
            "va": {},
            "pt": {},
            "delz": {},
            "pfull": {},
        }
        self.in_vars["parameters"] = ["bdt", "ptop"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "ua": {},
            "va": {},
            "pt": {},
            "delz": {}
        }
