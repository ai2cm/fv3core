import numpy as np

import fv3core.stencils.updatedzc as updatedzc
from fv3core.testing import TranslateFortranData2Py


class TranslateUpdateDzC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.update_gz_on_c_grid = updatedzc.UpdateGeopotentialHeightOnCGrid(grid)

        def compute(**kwargs):
            kwargs["dt"] = kwargs.pop("dt2")
            self.update_gz_on_c_grid(**kwargs)

        self.compute_func = compute
        self.in_vars["data_vars"] = {
            "dp_ref": {"serialname": "dp0"},
            "zs": {},
            "ut": {"serialname": "utc"},
            "vt": {"serialname": "vtc"},
            "gz": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "gz": grid.default_buffer_k_dict(),
            "ws": {"kstart": -1, "kend": None},
        }

    def compute(self, inputs):
        self.setup(inputs)
        outputs = self.slice_output(self.compute_from_storage(inputs))
        outputs["ws"] = self.subset_output("ws", outputs["ws"])
        # outputs["gz"] = self.subset_output("gz", outputs["gz"])
        return outputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        return self.update_gz_on_c_grid.subset_output(varname, output)
