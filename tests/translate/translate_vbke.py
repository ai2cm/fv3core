from .translate import TranslateFortranData2Py
from fv3core.stencils.vbke import vbke


class TranslateVbKE(TranslateFortranData2Py):
    def _call(self, *args, **kwargs):
        vbke(
            *args,
            **kwargs,
            cosa=self.grid.cosa,
            rsina=self.grid.rsina,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(
                self.grid.ie - self.grid.is_ + 2,
                self.grid.je - self.grid.js + 2,
                self.grid.npz,
            ),
            splitters=self.grid.splitters,
        )

    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self._call
        self.in_vars["data_vars"] = {
            "vc": {},
            "uc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}
