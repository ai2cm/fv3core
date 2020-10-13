from fv3core.stencils.circulation_cgrid import circulation_cgrid

from .translate import TranslateFortranData2Py


class TranslateCirculation_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }
        self.out_vars = {
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            }
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        circulation_cgrid(
            inputs["uc"],
            inputs["vc"],
            self.grid.dxc,
            self.grid.dyc,
            inputs["vort_c"],
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
        )
        return self.slice_output({"vort_c": inputs["vort_c"]})
