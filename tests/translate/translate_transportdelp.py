from fv3core.stencils.transportdelp import transportdelp
import fv3core.utils.gt4py_utils as utils

from .translate import TranslateFortranData2Py


class TranslateTransportDelp(TranslateFortranData2Py):
    def _call(self, **kwargs):
        orig = (self.grid.is_ - 1, self.grid.js - 1, 0)
        delpc = utils.make_storage_from_shape(kwargs["delp"].shape, origin=orig)
        ptc = utils.make_storage_from_shape(kwargs["pt"].shape, origin=orig)
        wc = utils.make_storage_from_shape(kwargs["w"].shape, origin=orig)
        transportdelp(
            **kwargs,
            rarea=self.grid.rarea,
            delpc=delpc,
            ptc=ptc,
            wc=wc,
            origin=orig,
            domain=self.grid.domain_shape_compute_buffer_2d(add=(2, 2, 0)),
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
        self.out_vars = {"delpc": {}, "ptc": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc, wc = self._call(**inputs)
        return self.slice_output(
            inputs,
            {"delpc": delpc, "ptc": ptc},
        )
