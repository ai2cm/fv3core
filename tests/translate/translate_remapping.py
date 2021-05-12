import fv3core.stencils.remapping as remapping
from fv3core.testing import TranslateFortranData2Py


class TranslateRemapping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = remapping.compute
        self.in_vars["data_vars"] = {
            "tracers": {},
            "w": {},
            "u": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
            "dp1": {},
            "delp": {},
            "cappa": {},
            "q_con": {},
            "pkz": grid.compute_dict(),
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kaxis": 1,
                "kend": grid.npz,
            },
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "hs": {"serialname": "phis"},
            "ps": {},
            "wsd": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "omga": {},
            "te0_2d": {
                "serialname": "te_2d",
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            # column variables...
            "ak": {},
            "bk": {},
            "pfull": {},
        }
        self.in_vars["parameters"] = [
            "ptop",
            "akap",
            "zvir",
            "last_step",
            "consv_te",
            "mdt",
            "bdt",
            "do_adiabatic_init",
            "nq",
        ]
        self.out_vars = {}
        self.write_vars = ["wsd"]
        for k in [
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "tracers",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "te0_2d",
            "u",
            "v",
            "w",
            "ps",
            "omga",
            "ua",
            "va",
            "dp1",
        ]:
            self.out_vars[k] = self.in_vars["data_vars"][k]
        self.out_vars["ps"] = {"kstart": grid.npz, "kend": grid.npz}
        self.max_error = 2e-8
        self.near_zero = 3e-18
        self.ignore_near_zero_errors = {"q_con": True, "tracers": True}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["q_cld"] = inputs["tracers"]["qcld"]
        self.compute_func(**inputs)
        inputs.pop("q_cld")
        return self.slice_output(inputs)
