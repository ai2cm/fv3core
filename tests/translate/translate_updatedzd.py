from typing import Optional

import numpy as np

import fv3core._config as spec
from fv3core.stencils import d_sw
from fv3core.stencils.updatedzd import UpdateHeightOnDGrid
from fv3core.testing import TranslateFortranData2Py


class TranslateUpdateDzD(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "dp0": {},  # column var
            "zs": {},
            "zh": {"kend": grid.npz + 1},
            "crx": grid.x3d_compute_domain_y_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "wsd": grid.compute_dict(),
        }

        self.in_vars["parameters"] = ["dt"]
        out_vars = ["zh", "crx", "cry", "xfx", "yfx", "wsd"]
        self.out_vars = {}
        for v in out_vars:
            self.out_vars[v] = self.in_vars["data_vars"][v]
        self.out_vars["wsd"]["kstart"] = grid.npz
        self.out_vars["wsd"]["kend"] = None
        self.updatedzd: Optional[UpdateHeightOnDGrid] = None

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.updatedzd = UpdateHeightOnDGrid(
            self.grid,
            spec.namelist,
            inputs.pop("dp0"),
            d_sw.get_column_namelist(),
            d_sw.k_bounds(),
        )
        inputs["x_area_flux"] = inputs.pop("xfx")
        inputs["y_area_flux"] = inputs.pop("yfx")
        inputs["surface_height"] = inputs.pop("zs")
        inputs["height"] = inputs.pop("zh")
        inputs["courant_number_x"] = inputs.pop("crx")
        inputs["courant_number_y"] = inputs.pop("cry")
        inputs["ws"] = inputs.pop("wsd")
        self.updatedzd(**inputs)
        inputs["xfx"] = inputs.pop("x_area_flux")
        inputs["yfx"] = inputs.pop("y_area_flux")
        inputs["zh"] = inputs.pop("height")
        inputs["crx"] = inputs.pop("courant_number_x")
        inputs["cry"] = inputs.pop("courant_number_y")
        inputs["wsd"] = inputs.pop("ws")
        outputs = self.slice_output(inputs)
        outputs["zh"] = self.subset_output("zh", outputs["zh"])
        return outputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname == "zh":
            if self.updatedzd is not None:
                output = output[self.updatedzd._zh_validator.validation_slice]
            else:
                raise RuntimeError("must call compute before calling subset_output")
        return output
