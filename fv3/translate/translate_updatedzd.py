from .translate import TranslateFortranData2Py
import fv3.stencils.updatedzd as updatedzd
from .translate_d_sw import TranslateD_SW


class TranslateUpdateDzD(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = updatedzd.compute
        self.in_vars["data_vars"] = {
            "ndif": {},  # column var
            "damp_vtd": {},  # column var
            "dp0": {},  # column var
            "zs": {},
            "zh": {'kend': grid.npz + 1},
            "crx": grid.x3d_compute_domain_y_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "wsd": grid.compute_dict()
        }
        #self.in_vars["data_vars"]["qe1_2"]["kaxis"] = 1
        #self.in_vars["data_vars"]["gam_2"]["kaxis"] = 1
        self.in_vars["parameters"] = ['dt']
        out_vars = ['zh', 'crx', 'cry', 'xfx', 'yfx', 'wsd']
        self.out_vars = {}
        for v in out_vars:
            self.out_vars[v] = self.in_vars["data_vars"][v]
        self.out_vars['wsd']['kstart'] = grid.npz
        self.out_vars['wsd']['kend'] = None
    
   
