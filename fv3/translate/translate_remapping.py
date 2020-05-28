from .translate import TranslateFortranData2Py
import fv3.stencils.remapping as remapping
import fv3.utils.gt4py_utils as utils


class TranslateRemapping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = remapping.compute
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "w": {},
            "u": grid.y3d_domain_dict(),
            "ua": {}, "va":{},
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
            "dp1": {},
            "delp": {},
            "cappa":{},
            "q_con": {},
            "pkz": grid.compute_dict(),
            "pk": {"istart": grid.is_, "iend": grid.ie, "jstart": grid.js, "jend": grid.je,  "kend": grid.npz},
            "peln": {"istart": grid.is_, "iend": grid.ie, "jstart": grid.js,"jend": grid.je,  "kaxis": 1, "kend": grid.npz},
            "pe": {"istart": grid.is_ - 1, "iend": grid.ie + 1, "jstart": grid.js - 1, "jend": grid.je + 1,
                   "kend": grid.npz + 1,"kaxis": 1},
            "hs": {"serialname":"phis"},
            "ps": {},
            "wsd": {"istart": grid.is_, "iend": grid.ie, "jstart": grid.js,"jend": grid.je},
            "omga": {},
            "te0_2d": {"serialname": "te_2d",
                       "istart": grid.is_, "iend": grid.ie, "jstart": grid.js, "jend": grid.je,
                       "kstart": grid.npz - 1, "kend": grid.npz - 1,
            },
            # column variables...
            "ak":{}, "bk":{}, "pfull": {},
           
           
        }
        self.in_vars["parameters"] = ["ptop", "akap", "zvir", "last_step", "consv_te", "mdt", "bdt", "kord_tracer", "do_adiabatic_init"]
        self.out_vars = {}
        for k in ["pe", "pkz","pk","peln","pt", "qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel", "qcld", "cappa", "delp", "delz", "q_con", "te0_2d", "u", "v", "w", "ps", "omga", "ua", "va", "dp1"]:
            self.out_vars[k] = self.in_vars["data_vars"][k]
        
