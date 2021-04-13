import fv3core.stencils.updatedzd
from fv3core.stencils import d_sw
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
        self.updatedzd = fv3core.stencils.updatedzd.UpdateDeltaZOnDGrid(
            self.grid, d_sw.get_column_namelist(), d_sw.k_bounds()
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.updatedzd(**inputs)
        self.out_vars["zh"].update(self.updatedzd._zh_validator.translate_dict)
        outputs = self.slice_output(inputs)
        outputs["zh"] = self.subset_output("zh", outputs["zh"])
        return outputs

    def subset_output(self, varname, output):
        if varname == "zh":
            output = output[self.updatedzd._zh_validator.validation_slice]
        return output
