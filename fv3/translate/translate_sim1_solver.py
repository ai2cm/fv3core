from .translate import TranslateFortranData2Py
import fv3.stencils.sim1_solver as sim1_solver
import fv3.utils.gt4py_utils as utils
import numpy as np


class TranslateSIM1_solver(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = sim1_solver.solve
        in_fields = ['pe2', 'dz2', 'w2', 'gm2', 'cp2', 'dm', 'pm2', 'pem', 'ptr', 'wsr']
        self.in_vars['data_vars'] = {}
        for f in in_fields:
            self.in_vars['data_vars'][f] = {'astart': grid.is_ - 1}
        self.in_vars['parameters'] = ['dt', 'gama', 'p_fac', 'akap']
        self.out_vars = {'pe2':  {'istart': grid.is_ - 1, 'iend': grid.ie + 1, 'kstart': 0, 'kend': grid.npz},
                         'dz2':  {'istart': grid.is_ - 1, 'iend': grid.ie + 1, 'kstart': 0, 'kend': grid.npz - 1},
                         'w2':   {'istart': grid.is_ - 1, 'iend': grid.ie + 1, 'kstart': 0, 'kend': grid.npz - 1},
        }

    def make_2d_storage_data(self, array2d, shape2d, astart=0, bstart=0):
        return utils.make_2d_storage_data(array2d, shape2d, istart=astart, jstart=bstart, origin=(astart, bstart, 0), backend=utils.data_backend)
        
     
    def compute(self, inputs):
        print('here')
        shape2d=self.grid.domain2d_ik_buffer_1cell()
        for d, info in self.in_vars['data_vars'].items():
            print(d, info, inputs[d].shape)
            if d == 'ptr':
                inputs[d] = inputs[d][:,0,:]
            arr = np.squeeze(inputs[d])
            if len(arr.shape) == 1:
                arr = np.repeat(arr[:, np.newaxis], shape2d[1], axis=1)
            inputs[d] = self.make_2d_storage_data(arr,
                                                  shape2d, astart=info['astart'])
        print(inputs.keys())
        #self.make_storage_data_input_vars(inputs)
        
        self.compute_func(**inputs)
        out = {}
        for var in self.out_vars.keys():
            info = self.out_vars[var]
            self.update_info(info, inputs)   
            ds = self.grid.default_domain_dict()
            ds.update(info)
            out[var] = np.squeeze(inputs[var].data[(slice(ds['istart'], ds['iend'] + 1), slice(ds['kstart'], ds['kend'] + 1))])
        return out
