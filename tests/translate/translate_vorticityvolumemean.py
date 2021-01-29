from fv3core.stencils.d_sw import vorticity_from_winds
from fv3core.testing import TranslateFortranData2Py


class TranslateVorticityVolumeMean(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"u": {}, "v": {}, "wk": {}}
        self.out_vars = {
            "wk": {},
            # "ut": grid.x3d_domain_dict(),
            # "vt": grid.y3d_domain_dict(),
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        vorticity_from_winds(
            inputs["u"],
            inputs["v"],
            self.grid.dx,
            self.grid.dy,
            self.grid.rarea,
            inputs["wk"],
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        return self.slice_output(inputs)
