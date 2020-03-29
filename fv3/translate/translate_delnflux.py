import fv3.stencils.delnflux as delnflux
from .translate_d_sw import TranslateD_SW
import fv3.stencils.d_sw as d_sw
import itertools

class TranslateDelnFlux(TranslateD_SW):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "q": {},
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
            "damp_c": {},
            "nord_column": {},
            "mass": {},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {"fx": grid.x3d_compute_dict(), "fy": grid.y3d_compute_dict()}

    # If use_sg is defined -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        if "mass" not in inputs:
            inputs["mass"] = None
        return self.nord_column_split_compute(inputs, delnflux.compute_delnflux_no_sg)
    
    def nord_column_split_compute(self, inputs, func):
        #[(x,y) for x in a for y in b]
        #list(itertools.product(a,b,c))
        return self.column_split_compute(
            inputs, func, {"nord": "nord_column", "damp_c": "damp_c"}
        )
    
    def column_split_compute(self, inputs, func, info_mapping):
        column_info = {}
        for pyfunc_var, serialbox_var in info_mapping.items():
            column_info[pyfunc_var] = self.column_namelist_vals(serialbox_var, inputs)
        self.make_storage_data_input_vars(inputs)
        for k in info_mapping.values():
            del inputs[k]
        kstarts = self.get_kstarts(column_info, self.grid)
        self.k_split_run(func, inputs, kstarts, column_info, self.grid)
        return self.slice_output(inputs)

    def get_kstarts(self, column_info, grid):
        compare = None
        kstarts = []
        for k in range(grid.npz):
            column_vals = {}
            for q, v in column_info.items():
                if k < len(v):
                    column_vals[q] = v[k]
            if column_vals != compare:
                kstarts.append(k)
            compare = column_vals
        for i in range(len(kstarts) - 1):
            kstarts[i] = (kstarts[i], kstarts[i + 1] - kstarts[i])
        kstarts[-1] = (kstarts[-1], grid.npz - kstarts[-1])
        print(kstarts)
        return kstarts
    def k_split_run(self, func, data, k_indices, splitvars_values, grid):
        for ki, nk in k_indices:
            splitvars = {}
            for name, value_array in splitvars_values.items():
                splitvars[name] = value_array[ki]
            data.update(splitvars)
            data['kstart'] = ki
            data['nk'] = nk
            func(**data)

class TranslateDelnFlux_2(TranslateDelnFlux):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars["data_vars"]["mass"]
