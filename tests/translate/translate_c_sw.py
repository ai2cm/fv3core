import fv3core.stencils.c_sw as c_sw
import fv3core.utils.gt4py_utils as utils

from .translate import TranslateFortranData2Py


class TranslateC_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = c_sw.compute
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "u": {"jend": grid.jed + 1},
            "v": {"iend": grid.ied + 1},
            "w": {},
            "uc": {"iend": grid.ied + 1},
            "vc": {"jend": grid.jed + 1},
            "ua": {},
            "va": {},
            "ut": {},
            "vt": {},
            "omga": {"serialname": "omgad"},
            "divgd": {"iend": grid.ied + 1, "jend": grid.jed + 1},
        }
        self.in_vars["parameters"] = ["dt2"]
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.out_vars = {}
        for v, d in self.in_vars["data_vars"].items():
            self.out_vars[v] = d
        for servar in ["delpcd", "ptcd"]:
            self.out_vars[servar] = {}
        # TODO - fix edge_interpolate4 in d2a2c_vect to match closer and the variables here should as well
        self.max_error = 2e-10

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc = self.compute_func(**inputs)
        return self.slice_output(inputs, {"delpcd": delpc, "ptcd": ptc})


class TranslateTransportDelp(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "utc": {},
            "vtc": {},
            "w": {},
            "wc": {},
        }
        self.out_vars = {"delpc": {}, "ptc": {}, "wc": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        orig = (self.grid.is_ - 1, self.grid.js - 1, 0)
        delpc = utils.make_storage_from_shape(inputs["delp"].shape, origin=orig)
        ptc = utils.make_storage_from_shape(inputs["pt"].shape, origin=orig)
        wc = utils.make_storage_from_shape(inputs["w"].shape, origin=orig)
        c_sw.transportdelp(
            **inputs,
            rarea=self.grid.rarea,
            delpc=delpc,
            ptc=ptc,
            origin=orig,
            domain=self.grid.domain_shape_compute_buffer_2d(add=(2, 2, 0)),
        )
        return self.slice_output(
            inputs,
            {"delpc": delpc, "ptc": ptc, "wc": wc},
        )


class TranslateCirculation_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }
        self.out_vars = {
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            }
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        c_sw.circulation_cgrid(
            inputs["uc"],
            inputs["vc"],
            self.grid.dxc,
            self.grid.dyc,
            inputs["vort_c"],
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
        )
        return self.slice_output({"vort_c": inputs["vort_c"]})
