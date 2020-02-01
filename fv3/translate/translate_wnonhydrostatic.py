from .translate import TranslateFortranData2Py
from fv3.stencils.d_sw import flux_adjust

class TranslateWnonhydrostatic(TranslateFortranData2Py):
    def __init__(self, grid):
       
        super().__init__(grid)
        self.in_vars['data_vars'] = {'w': {}, 'delp': {}, 'gx': {}, 'gy': {}}
        self.out_vars = {'w': {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        flux_adjust(inputs['w'], inputs['delp'], inputs['gx'], inputs['gy'],
                    self.grid.rarea, origin=self.grid.compute_origin(),
                    domain=self.grid.domain_shape_compute())
        return self.slice_output(inputs)
