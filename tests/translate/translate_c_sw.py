import fv3core._config as spec
from fv3core.stencils.c_sw import CGridShallowWaterDynamics
from fv3core.testing import TranslateFortranData2Py


class TranslateC_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        cgrid_shallow_water_lagrangian_dynamics = CGridShallowWaterDynamics(
            grid, spec.namelist
        )
        self.compute_func = cgrid_shallow_water_lagrangian_dynamics
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
        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-10

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        outputs = {"delpcd": self.compute_func.delpc, "ptcd": self.compute_func.ptc}
        return self.slice_output(inputs, outputs)


class TranslateVorticityTransport_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        cgrid_sw_lagrangian_dynamics = CGridShallowWaterDynamics(grid, spec.namelist)
        self.compute_func = cgrid_sw_lagrangian_dynamics._vorticitytransport_cgrid
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "ke_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "u": {},
            "v": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}
