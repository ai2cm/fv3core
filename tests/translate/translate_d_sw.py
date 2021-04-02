from gt4py.gtscript import PARALLEL, computation, interval

from fv3core.decorators import gtstencil
from fv3core.stencils import d_sw
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.typing import FloatField, FloatFieldIJ


class TranslateD_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.max_error = 3e-11  # propagated error from vt roundoff error in FxAdv
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

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        for v in [
            "utco",
            "vtco",
            "keco",
            "uco",
            "uvort",
            "kex",
            "kevort",
            "vco",
            "ubkey",
            "vbkey",
            "fyh",
            "uts",
            "utafter",
        ]:
            if v in inputs:
                del inputs[v]

        d_sw.compute(**inputs)
        return self.slice_output(inputs)


class TranslateUbKE(TranslateFortranData2Py):
    @gtstencil()
    def ubke(
        uc: FloatField,
        vc: FloatField,
        cosa: FloatFieldIJ,
        rsina: FloatFieldIJ,
        ut: FloatField,
        ub: FloatField,
        dt4: float,
        dt5: float,
    ):
        with computation(PARALLEL), interval(...):
            ub = d_sw.ubke(uc, vc, cosa, rsina, ut, ub, dt4, dt5)

    def _call(self, *args, **kwargs):
        self.ubke(
            *args,
            **kwargs,
            cosa=self.grid.cosa,
            rsina=self.grid.rsina,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self._call
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": {},
            "ub": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"ub": grid.compute_dict_buffer_2d()}


class TranslateVbKE(TranslateFortranData2Py):
    @gtstencil()
    def vbke(
        vc: FloatField,
        uc: FloatField,
        cosa: FloatFieldIJ,
        rsina: FloatFieldIJ,
        vt: FloatField,
        vb: FloatField,
        dt4: float,
        dt5: float,
    ):
        with computation(PARALLEL), interval(...):
            vb = d_sw.vbke(vc, uc, cosa, rsina, vt, vb, dt4, dt5)

    def _call(self, *args, **kwargs):
        self.vbke(
            *args,
            **kwargs,
            cosa=self.grid.cosa,
            rsina=self.grid.rsina,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self._call
        self.in_vars["data_vars"] = {
            "vc": {},
            "uc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}


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
