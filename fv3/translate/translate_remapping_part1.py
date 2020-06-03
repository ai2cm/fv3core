from .translate import TranslateFortranData2Py
import fv3.stencils.remapping_part1 as remap_part1
import fv3.utils.gt4py_utils as utils


class TranslateRemapping_Part1(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = remap_part1.compute
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
            "ua": {},
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
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
            "hs": {},
            "te": {},
            "ps": {},
            "wsd": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "omga": {},
            # column variables...
            "ak": {},
            "bk": {},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
        }
        self.in_vars["parameters"] = [
            "ptop",
            "akap",
            "r_vir",
        ]
        self.out_vars = {}
        for k in [
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "qvapor",
            "qliquid",
            "qice",
            "qrain",
            "qsnow",
            "qgraupel",
            "qcld",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "te",
            "u",
            "v",
            "w",
            "ps",
            "wsd",
            "omga",
            "ua",
        ]:
            self.out_vars[k] = self.in_vars["data_vars"][k]
        self.out_vars["wsd"]["kstart"] = grid.npz - 1
        self.out_vars["wsd"]["kend"] = grid.npz - 1
        self.out_vars["ps"] = {'kstart': grid.npz - 1, "kend": grid.npz - 1}
        self.max_error = 1e-12
        # column_var = {'istart': grid.is_, 'iend': grid.ie, 'jstart': grid.js, 'jend': grid.je, "kend": grid.npz + 1}
        # self.out_vars['ak'] = column_var
        # self.out_vars['bk'] = column_var
