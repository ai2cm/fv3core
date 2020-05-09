from .translate import TranslateFortranData2Py
import fv3.stencils.ubke as ubke


class TranslateNeg_Adj3(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = ubke.compute
        self.in_vars["data_vars"] = {
        }
        self.in_vars["parameters"] = ["cld_amt"]
        self.out_vars = {}
    def compute(self, inputs):
        print('to cloud or not to cloud', inputs['cld_amt'])
        return []
