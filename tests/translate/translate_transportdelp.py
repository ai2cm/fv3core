from fv3core.stencils.transportdelp import transportdelp

from .translate import TranslateFortranData2Py


class TranslateTransportDelp(TranslateFortranData2Py):
    def _call(self, *args, **kwargs):
        transportdelp(
            *args,
            **kwargs,
            origin=(self.grid.is_ - 1, self.grid.js - 1, 0),
            domain=self.grid.domain_shape_compute_buffer_2d(add=(2, 2, 0)),
            splitters=self.grid.splitters,
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "w": {},
            "utc": {},
            "vtc": {},
            "wc": {},
            "delpc": {}, "ptc": {},
        }
        self.out_vars = {"delpc": {}, "ptc": {}, "wc": {}}

    def compute(self, storages):
        self.make_storage_data_input_vars(storages)
        self._call(**storages)
        return self.slice_output(storages, {"delpc": storages["delpc"], "ptc": storages["ptc"]})
