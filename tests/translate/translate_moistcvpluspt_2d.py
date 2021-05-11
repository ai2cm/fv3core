import fv3core.stencils.moist_cv as moist_cv
from fv3core.testing import TranslateFortranData2Py, TranslateGrid, pad_field_in_j
from fv3core.utils.typing import FloatField
from fv3core.decorators import FrozenStencil

def moist_pt(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: float,
):
    with computation(PARALLEL), interval(...):
        cvm, gz, q_con, cappa, pt = moist_cv.moist_pt_func(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            gz,
            cvm,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )

class TranslateMoistCVPlusPt_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname":"qice_js"},
            "qrain": {"serialname":"qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname":"qgraupel_js"},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
        }
        self.write_vars = ["gz", "cvm"]
        for k, v in self.in_vars["data_vars"].items():
            if k not in self.write_vars:
                v["axis"] = 1

        self.in_vars["parameters"] = ["r_vir"]
        self.out_vars = {
            "gz": {
                "serialname": "gz1d",
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "cvm": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "pt": {},
            "cappa": {},
            "q_con": {},
        }

    def compute_from_storage(self, inputs):
        moist_cv_pt = FrozenStencil(
            moist_pt,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(self.grid.nic, 1, self.grid.npz)
        )
        for name, value in inputs.items():
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(value, self.grid.njd)
                )
        moist_cv_pt(**inputs)
        return inputs
