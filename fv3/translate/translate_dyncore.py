from .parallel_translate import ParallelTranslate2Py
from .translate import TranslateFortranData2Py
import fv3.stencils.dyn_core as dyn_core
import fv3util


class TranslateDynCore(ParallelTranslate2Py):
    inputs = {
        "q_con": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "pt": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "K",
        },
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
    }
    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = dyn_core.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "cappa": {},
            "u" : grid.y3d_domain_dict(),
            "v" : grid.x3d_domain_dict(),
            "w" : {},
            "delz" : {},
            "delp": {},
            "pt": {},
            #  "q4d": {},
            "pe" : {'istart': grid.is_ - 1, 'iend': grid.ie + 1,
                    "jstart": grid.js - 1, "jend": grid.je + 1,
                    "kend": grid.npz + 1, "kaxis": 1,},
            "pk":  {"istart": grid.is_, "iend": grid.ie, "jstart": grid.js, "jend": grid.je, "kend": grid.npz + 1},
            "phis": {'kstart': 0, 'kend':0},
            "wsd": grid.compute_dict(),
            "omga": {},
            "ua" : {},
            "va" : {},
            "uc" : grid.x3d_domain_dict(),
            "vc" : grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "pkz": grid.compute_dict(),
            "peln": {"istart": grid.is_, "iend": grid.ie, "jstart": grid.js, "jend": grid.je, "kend": grid.npz + 1, "kaxis": 1},
            "q_con": {},
            "ak": {},
            "bk": {},
            "diss_estd": {},
            
            
           
        }
        self._base.in_vars['data_vars']['wsd']['kstart'] = grid.npz
        self._base.in_vars['data_vars']['wsd']['kend'] = None
        # bad uc, vc, gz, ws3, divgd
        # bad corners: omgad, delpc, ptc, ut, vt
        # ut and vt are REALLY bad, vt also bad at 9, 9
        # small non edge errors -- divgd
        # super wrong uc, vc
        # working delp,pt, ua, va, u, v, w
        self._base.in_vars["parameters"] = ["mdt", "n_split", "akap", "ptop", "pfull", "n_map_step"]
        self._base.out_vars = {}
        #self.out_vars = { "cappa": {}}
        for v, d in self._base.in_vars["data_vars"].items():
            self._base.out_vars[v] = d
        del self._base.out_vars['ak']
        del self._base.out_vars['bk']
        '''
        self.in_vars["data_vars"].update({"ptd_a": {}, 'ud_a':  grid.y3d_domain_dict(), 'vd_a': grid.x3d_domain_dict(), 'ucd_a':{}, 'vcd_a':{}, 'uad_a': {}, 'vad_a':{}, 'utd_a':{}, 'vtd_a':{}, 'divgdd_a':{}, 'wd_a':{},'delpd_a':{}, 'delpcd_a':{}, 'ptcd_a': {},
            'ucd_b':{}, 'vcd_b':{}, 'divgdd_b':{}, 'omgad_b':{}, 'gzd_b':{}, 'delpcd_b':{}, 'pkcd_b':{}, 'cappad_b':{}, 'phisd_b':{}, 'ptcd_b':{}, 'q_cond_b':{}, 'ws3d_b':{}, 'dp_refd_b':{}, 'zsd_b':{}, 'utd_b':{}, 'vtd_b':{}, 'zhd_b':{},
            'ucd_3':{}, 'ucd_2': {}, 'vcd_2':{}, 'vcd_3':{},'gzd_4':{},'pkcd_3':{}, 'gzd_3':{}, 'gzd_r':{}, 'ws3d_r':{}, 'gzd_u':{}, 'dp_refd_u':{}, 'utd_u':{}, 'vtd_u':{}, 'zsd_u':{}, 'ws3d_u':{},'gzd_n': {}, 'gzd_h':{},
                                          'vtd_d': {},'delpd_d': {},'ptcd_d': {},'ptd_d': {}, 'ud_d': {}, 'vd_d': {}, 'wd_d': {},'ucd_d': {}, 'vcd_d': {},'uad_d': {},'vad_d': {},'divgdd_d': {}, 'mfxdd_d': grid.x3d_compute_dict(), 'mfydd_d': grid.y3d_compute_dict(),
                                          'cxdd_d': grid.x3d_compute_domain_y_dict(),'cydd_d': grid.y3d_compute_domain_x_dict(),
                                          'crxd_d': grid.x3d_compute_domain_y_dict(),'cryd_d': grid.y3d_compute_domain_x_dict(),
                                          'xfxd_d':  grid.x3d_compute_domain_y_dict(),'yfxd_d': grid.y3d_compute_domain_x_dict(),
                                          'q_cond_d': {}, 'heat_sourced_d': {}, 'diss_estdd_d':{},
                                          'q_cond_h2':{}, 'delpd_h2':{}, 'ptd_h2':{},
                                          'zhd_z':{}, 'crxd_z':grid.x3d_compute_domain_y_dict(), 'cryd_z': grid.y3d_compute_domain_x_dict(), 'xfxd_z': grid.x3d_compute_domain_y_dict(), 'yfxd_z': grid.y3d_compute_domain_x_dict(), 'delzd_z':{}, 'wsdd_z':grid.compute_dict(),
                                          'pk3d_h3':{}, 'ped_h3':{"istart": grid.is_ - 1,"jstart": grid.js - 1,"kaxis": 1,},
                                          'zhd_h4': {}, 'wd_s':{},'delzd_s':{},'zhd_s':{},
                                          'ped_s':{"istart": grid.is_ - 1,"jstart": grid.js - 1,"kaxis": 1,},'pkcd_s':{},'pk3d_s':{}, 'pkd_s':{}, 'pelnd_s':{"istart": grid.is_, "jstart": grid.js, "kaxis": 1}, 'wsdd_s':{"istart": grid.is_, "jstart": grid.js},

                                          'vtd_lol': {}, 'delpd_lol': {}, 'ptcd_lol': {}, 'ptd_lol': {},  'ud_lol':{},'vd_lol': {}, 'wd_lol': {}, 'ucd_lol': {}, 'vcd_lol': {}, 'uad_lol': {}, 'vad_lol': {}, 'divgdd_lol': {}, 'mfxdd_lol':  grid.x3d_compute_dict(), 'mfydd_lol': grid.y3d_compute_dict(), 'cxdd_lol':  grid.x3d_compute_domain_y_dict(), 'cydd_lol':  grid.y3d_compute_domain_x_dict(), 'crxd_lol':  grid.x3d_compute_domain_y_dict(), 'cryd_lol':  grid.y3d_compute_domain_x_dict(), 'xfxd_lol':  grid.x3d_compute_domain_y_dict(), 'yfxd_lol':  grid.y3d_compute_domain_x_dict(), 'q_cond_lol': {}, 'zhd_lol': {},
                                          'zsd_9':{}, 'wd_9':{}, 'delzd_9':{}, 'q_cond_9':{}, 'delpd_9':{}, 'ptd_9':{}, 'zhd_9':{}, 'ped_9':{"istart": grid.is_-1, "jstart": grid.js-1, "kaxis": 1}, 'pkcd_9':{}, 'pk3d_9':{}, 'pkd_9':{}, 'pelnd_9':{"istart": grid.is_, "jstart": grid.js, "kaxis": 1}, 'wsdd_9':{"istart": grid.is_, "jstart": grid.js},
        })
        '''
        # TODO - fix edge_interpolate4 in d2a2c_vect to match closer and the variables here should as well
        self.max_error = 1e-6
    '''   
    def compute_sequential(self, inputs_list, communicator_list):
        self._base.make_storage_data_input_vars(inputs_list)
        self.make_storage_data_input_vars(inputs_list)
        dyn_core.compute(inputs_list, communicator_list)
        outputs = []
        for ins in inputs_list:
            outputs.append(self.slice_output(ins))
        return outputs
       
    '''
    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        for name, properties in self.inputs.items():
            inputs[name + '_quantity'] = self.grid.quantity_wrap(inputs[name], dims=properties['dims'], units=properties['units'])
        state = {'data': inputs, 'comm': communicator}
        self._base.compute_func(**state)
        return self._base.slice_output(state['data'])
