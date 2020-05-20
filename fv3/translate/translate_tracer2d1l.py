from .translate import TranslateFortranData2Py
import fv3.stencils.tracer_2d_1l as tracer_2d_1l


class TranslateTracer2D1L(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = tracer_2d_1l.compute
        self.in_vars["data_vars"] = {
            "qvapor" :{},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "dp1": {},
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
        }
        self.in_vars["parameters"] = ["nq", "q_split", "mdt"]
        # q_split is a namelist var
        self.out_vars = self.in_vars["data_vars"]
