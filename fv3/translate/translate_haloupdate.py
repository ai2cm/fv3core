from .translate import TranslateFortranData2Py
from fv3.stencils.haloupdate import halo_update
import fv3._config as spec
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class TranslateHeatDiss(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "qcon": {},
        }
        self.in_vars["parameters"] = ["dd8"]
        self.out_vars = {
            "delp": grid.compute_dict(),
            "pt": grid.compute_dict(),
            "qcon": grid.compute_dict(),
        }

    def compute(self, inputs):
        if MPI is None:
            raise ImportError('could not import mpi4py')
        self.make_storage_data_input_vars(inputs)
        for value in inputs.values():
            halo_update(value, spec.namelist['layout'])
        return self.slice_output(inputs)
