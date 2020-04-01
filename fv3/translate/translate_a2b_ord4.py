import fv3.stencils.divergence_damping as dd
from fv3.translate.translate import TranslateFortranData2Py


class TranslateA2B_Ord4(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = dd.vorticity_calc
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"wk": {}, "vort": {}}
        self.compute_func = dd.vorticity_calc

    def compute(self, inputs):
        return self.column_split_compute(inputs, {"nord": "nord_col"})
