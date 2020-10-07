from fv3core.stencils.transportdelp import transportdelp
import fv3core.utils.gt4py_utils as utils

from .translate import TranslateFortranData2Py


class TranslateTransportDelp(TranslateFortranData2Py):
    def _call(self, *args, **kwargs):
        orig = (self.grid.is_ - 1, self.grid.js - 1, 0)
        delpc = utils.make_storage_from_shape(kwargs["delp"].shape, origin=orig)
        ptc = utils.make_storage_from_shape(kwargs["pt"].shape, origin=orig)
        wc = utils.make_storage_from_shape(kwargs["w"].shape, origin=orig)
        transportdelp(
            *args,
            **kwargs,
            rarea=self.grid.rarea,
            delpc=delpc,
            ptc=ptc,
            wc=wc,
            origin=(self.grid.is_ - 1, self.grid.js - 1, 0),
            domain=self.grid.domain_shape_compute_buffer_2d(add=(2, 2, 0)),
            splitters=self.grid.splitters,
        )
        return delpc, ptc, wc

    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "utc": {},
            "vtc": {},
            "w": {},
        }
        self.out_vars = {"delpc": {}, "ptc": {}, "wc": {}}

    def compute(self, storages):
        self.make_storage_data_input_vars(storages)
        delpc, ptc, wc = self._call(**storages)
        return self.slice_output(
            storages,
            {"delpc": delpc, "ptc": ptc, "wc": wc},
        )
