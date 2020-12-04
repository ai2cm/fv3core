import fv3core.stencils.dyn_core as dyn_core
import fv3gfs.util as fv3util

from .parallel_translate import ParallelTranslate2PyState


class TranslateDynCore(ParallelTranslate2PyState):
    inputs = {
        "q_con": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "pt": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "K"},
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m/s"},
    }

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = dyn_core.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "cappa": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "delp": {},
            "pt": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "phis": {"kstart": 0, "kend": 0},
            "wsd": grid.compute_dict(),
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "pkz": grid.compute_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "q_con": {},
            "ak": {},
            "bk": {},
            "diss_estd": {},
            "pfull": {},
        }
        self._base.in_vars["data_vars"]["wsd"]["kstart"] = grid.npz
        self._base.in_vars["data_vars"]["wsd"]["kend"] = None

        self._base.in_vars["parameters"] = [
            "mdt",
            "n_split",
            "akap",
            "ptop",
            "n_map",
            "ks",
        ]
        self._base.out_vars = {}

        for v, d in self._base.in_vars["data_vars"].items():
            self._base.out_vars[v] = d
        del self._base.out_vars["ak"]
        del self._base.out_vars["bk"]

        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-6
