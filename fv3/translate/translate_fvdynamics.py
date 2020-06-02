from .parallel_translate import ParallelTranslate2Py
import fv3.stencils.fv_dynamics as fv_dynamics


class TranslateFVDYnamics(ParallelTranslate2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = fv_dynamics.compute
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "w": {},
            "delz": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "ps": {},
            "pe": {},
            "pk": {},
            "peln": {},
            "pkz": {},
            "phis": {},
            "q_con": {},
            "omga": {},
            "ua": {},
            "va": {},
            "uc": {},
            "vc":{},
            "ak":{},
            "bk": {},
            "mfxd":{},
            "mfyd":{},
            "cxd":{},
            "cyd":{},
            "diss_estd": {}
        }
        self.in_vars["parameters"] = ["bdt", "zvir", "ptop", "ks", "n_split"]
        self.out_vars = self.in_vars["data_vars"]
        for var in ['ak', 'bk']:
            del  self.out_vars[var]
