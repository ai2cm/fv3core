from typing import Optional

import fv3core._config as spec
from fv3core.stencils import d_sw
from fv3core.stencils.updatedzd import UpdateDeltaZOnDGrid
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
        self.updatedzd: Optional[UpdateDeltaZOnDGrid] = None

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["x_area_flux"] = inputs.pop("xfx")
        inputs["y_area_flux"] = inputs.pop("yfx")
        self.updatedzd = UpdateDeltaZOnDGrid(
            self.grid,
            spec.namelist,
            inputs.pop("dp0"),
            d_sw.get_column_namelist(),
            d_sw.k_bounds(),
        )
        self.updatedzd(**inputs)
        inputs["xfx"] = inputs.pop("x_area_flux")
        inputs["yfx"] = inputs.pop("y_area_flux")
        self.out_vars["zh"].update(self.updatedzd._zh_validator.translate_dict)
        outputs = self.slice_output(inputs)
        outputs["zh"] = self.subset_output("zh", outputs["zh"])
        return outputs

    def subset_output(self, varname, output):
        if varname == "zh":
            if self.updatedzd is not None:
                output = output[self.updatedzd._zh_validator.validation_slice]
            else:
                raise RuntimeError("must call compute before calling subset_output")
        return output
