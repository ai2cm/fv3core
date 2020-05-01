from .translate import TranslateFortranData2Py
import fv3.stencils.map_ppm_2d as Map1_PPM_2d


class TranslateMap1_PPM_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Map1_PPM_2d.compute
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "var_in"},
            "pe1": {},
            "pe2": {},
            "qs": {"serialname": "ws_1d"},
        }
        self.in_vars["parameters"] = {"jj": {"serialname":"j_2d"}, "i1":{}, "i2":{}, "iv":{"serialname": "mode"}, "kord":{}}
        self.out_vars = {
            "var_inout": {},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        var_inout = self.compute_func(**inputs)
        return self.slice_output(inputs, {"var_inout": var_inout})
