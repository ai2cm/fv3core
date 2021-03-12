import fv3core.stencils.d2a2c_vect as d2a2c_vect
from fv3core.testing import TranslateFortranData2Py


class TranslateD2A2C_Vect(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        self.in_vars["parameters"] = ["dord4"]
        self.out_vars = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        # TODO: This seems to be needed primarily for the edge_interpolate_4
        # methods, can we rejigger the order of operations to make it match to
        # more precision?
        self.max_error = 2e-10

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        # d2a2c_vect is written assuming dord4 is True
        assert bool(inputs["dord4"]) is True
        del inputs["dord4"]

        d2a2c_vect.compute(self.grid, **inputs)

        return self.slice_output(inputs)
