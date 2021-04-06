import fv3core.stencils.updatedzc as updatedzc
from fv3core.testing import TranslateFortranData2Py


class TranslateUpdateDzC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "dp_ref": {"serialname": "dp0"},
            "gz_surface": {"serialname": "zs"},
            "ut": {"serialname": "utc"},
            "vt": {"serialname": "vtc"},
            "gz": {},
            "surface_delta_gz": {"serialname": "ws"},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "gz": grid.default_buffer_k_dict(),
            "surface_delta_gz": {"serialname": "ws", "kstart": -1, "kend": None},
        }

    def compute_from_storage(self, inputs):
        updatedzc.update_dz_c_stencil(
            area=self.grid.area,
            **inputs,
            origin=self.grid.compute_origin(add=(-1, -1, 0)),
            domain=self.grid.domain_shape_compute(add=(2, 2, 1)),
        )
        return inputs
