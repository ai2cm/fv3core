import fv3core._config as spec
import fv3core.stencils.d_sw as d_sw
from fv3core.decorators import FrozenStencil
from fv3core.testing import TranslateFortranData2Py


class TranslateD_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.max_error = 6e-11
        column_namelist = d_sw.get_column_namelist(spec.namelist, grid.npz)
        self.compute_func = d_sw.DGridShallowWaterLagrangianDynamics(
            spec.namelist, column_namelist
        )
        self.in_vars["data_vars"] = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "w": {},
            "delpc": {},
            "delp": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "crx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "mfx": grid.x3d_compute_dict(),
            "mfy": grid.y3d_compute_dict(),
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "heat_source": {},
            "diss_est": {},
            "q_con": {},
            "pt": {},
            "ptc": {},
            "ua": {},
            "va": {},
            "zh": {},
            "divgd": grid.default_dict_buffer_2d(),
        }
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = self.in_vars["data_vars"].copy()
        del self.out_vars["zh"]


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
        self.compute_func = FrozenStencil(
            d_sw.flux_capacitor,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )


class TranslateHeatDiss(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "fx2": {},
            "fy2": {},
            "w": {},
            "dw": {},
            "heat_source": {},
            "diss_est": {},
        }
        self.out_vars = {
            "heat_source": grid.compute_dict(),
            "diss_est": grid.compute_dict(),
            "dw": grid.compute_dict(),
        }

    def compute_from_storage(self, inputs):
        column_namelist = d_sw.get_column_namelist(spec.namelist, self.grid.npz)
        # TODO add these to the serialized data or remove the test
        inputs["damp_w"] = column_namelist["damp_w"]
        inputs["ke_bg"] = column_namelist["ke_bg"]
        inputs["dt"] = (
            spec.namelist.dt_atmos / spec.namelist.k_split / spec.namelist.n_split
        )
        inputs["rarea"] = self.grid.rarea
        heat_diss_stencil = FrozenStencil(
            d_sw.heat_diss,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        heat_diss_stencil(**inputs)
        return inputs


class TranslateWdivergence(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"w": {}, "delp": {}, "gx": {}, "gy": {}}
        self.out_vars = {"w": {}}
        self.compute_func = FrozenStencil(
            d_sw.flux_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

    def compute_from_storage(self, inputs):
        inputs["rarea"] = self.grid.rarea
        self.compute_func(**inputs)
        return inputs
