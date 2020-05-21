from .parallel_translate import JustParallelTranslate
import fv3.stencils.tracer_2d_1l as tracer_2d_1l
import fv3util

class TranslateTracer2D1L(JustParallelTranslate):
    inputs = {
        "qvapor": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qliquid": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qice": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qrain": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qsnow": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qgraupel": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
        "qcld": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        },
    }
    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = tracer_2d_1l.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "qvapor" : {},      
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
        self._base.in_vars["parameters"] = ["nq", "mdt"]
        self._base.out_vars = self._base.in_vars["data_vars"]
'''
 "qv_a":{},"qv_b":{}, "qv_c":{},"dp1_c":{},
            "dp2_c":{"istart":grid.is_, "jstart": grid.js},"fx_c":{"istart":grid.is_, "jstart": grid.js}, "fy_c":{"istart":grid.is_, "jstart": grid.js},
            "xfx_b": {"istart":grid.is_},
            "yfx_b": {"jstart":grid.js},
            "cx_b": {"istart":grid.is_},
            "cy_b": {"jstart":grid.js},
            "rax_b":{"istart":grid.is_},
            "ray_b":{"jstart": grid.js},
'''
