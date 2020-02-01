import fv3.utils.gt4py_utils as utils
import numpy as np
import gt4py as gt
class Grid:
    indices = ['is_', 'ie', 'isd', 'ied', 'js', 'je', 'jsd', 'jed']
    shape_params = ['npz', 'npx', 'npy']
    # npx -- number of grid corners on one tile of the domain
    # grid.ie == npx - 1identified east edge in fortran
    # But we need to add the halo - 1 to change this check to 0 based python arrays
    # grid.ie == npx + halo - 2
    def __init__(self, indices, shape_params, data_fields={}):
        for i in self.indices:
            setattr(self, i, int(indices[i]))
        for s in self.shape_params:
            setattr(self, s, int(shape_params[s]))
        self.nid = int(self.ied - self.isd + 1)
        self.njd = int(self.jed - self.jsd + 1)
        self.nic = int(self.ie - self.is_ + 1)
        self.njc = int(self.je - self.js + 1)
        self.halo = utils.halo
        self.west_edge = self.is_ == self.halo
        self.east_edge = self.ie == self.npx + self.halo - 2
        self.south_edge = self.js == self.halo
        self.north_edge = self.je == self.npy + self.halo - 2
        self.j_offset = self.js - self.jsd - 1
        self.i_offset = self.is_ - self.isd - 1
        self.sw_corner = self.west_edge and self.south_edge
        self.se_corner = self.east_edge and self.south_edge
        self.nw_corner = self.west_edge and self.north_edge
        self.ne_corner = self.east_edge and self.north_edge
        self.data_fields = {}
        self.add_data(data_fields)

    def add_data(self, data_dict):
        self.data_fields.update(data_dict)
        for k, v in self.data_fields.items():
            setattr(self, k, v)
    
    def get_index(self, name):
        return getattr(self, name)

    def irange_compute(self):
        return range(self.is_, self.ie + 1)

    def irange_compute_x(self):
        return range(self.is_, self.ie + 2)

    def jrange_compute(self):
        return range(self.js, self.je + 1)

    def jrange_compute_y(self):
        return range(self.js, self.je + 2)

    def irange_domain(self):
        return range(self.isd, self.ied + 1)

    def jrange_domain(self):
        return range(self.jsd, self.jed + 1)

    def krange(self):
        return range(0, self.npz)

    def compute_interface(self):
        return self.slice_dict(self.compute_dict())
    
    def x3d_interface(self):
        return self.slice_dict(self.x3d_compute_dict())

    def y3d_interface(self):
        return self.slice_dict(self.y3d_compute_dict())

    def x3d_domain_interface(self):
        return self.slice_dict(self.x3d_domain_dict())

    def y3d_domain_interface(self):
        return self.slice_dict(self.y3d_domain_dict())

    def add_one(self, num):
        if num is None:
            return None
        return num + 1
    
    def slice_dict(self, d):
        return (slice(d['istart'], self.add_one(d['iend'])),
                slice(d['jstart'], self.add_one(d['jend'])),
                slice(d['kstart'], self.add_one(d['kend'])))

    def default_domain_dict(self):
        return {
            'istart': self.isd, 'iend': self.ied,
            'jstart': self.jsd, 'jend': self.jed,
            'kstart': 0, 'kend': self.npz-1,
        }
    
    def default_dict_buffer_2d(self):
        mydict = self.default_domain_dict()
        mydict['iend'] += 1
        mydict['jend'] += 1
        return mydict
    
    def compute_dict(self):
        return {
            'istart': self.is_, 'iend': self.ie,
            'jstart': self.js, 'jend': self.je,
            'kstart': 0, 'kend': self.npz-1,
        }
    def compute_dict_buffer_2d(self):
        mydict = self.compute_dict()
        mydict['iend'] += 1
        mydict['jend'] += 1
        return mydict
    
    def default_buffer_k_dict(self):
        mydict = self.default_domain_dict()
        mydict['kend'] = self.npz
        return mydict
    
    def x3d_domain_dict(self):
        horizontal_dict = {'istart': self.isd, 'iend': self.ied + 1,
                           'jstart': self.jsd, 'jend': self.jed}
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_domain_dict(self):
        horizontal_dict = {'istart': self.isd, 'iend': self.ied,
                           'jstart': self.jsd, 'jend': self.jed + 1}
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_dict(self):
        horizontal_dict = {'istart': self.is_, 'iend': self.ie + 1,
                           'jstart': self.js, 'jend': self.je}
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_dict(self):
        horizontal_dict = {'istart': self.is_, 'iend': self.ie,
                           'jstart': self.js, 'jend': self.je + 1}
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_domain_y_dict(self):
        horizontal_dict = {'istart': self.is_, 'iend': self.ie + 1,
                           'jstart': self.jsd, 'jend': self.jed}
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_domain_x_dict(self):
        horizontal_dict = {'istart': self.isd, 'iend': self.ied,
                           'jstart': self.js, 'jend': self.je + 1}
        return {**self.default_domain_dict(), **horizontal_dict}

    def domain_shape_standard(self):
        return (self.nid, self.njd, self.npz)

    def domain_shape_buffer_k(self):
        return (self.nid, self.njd, self.npz + 1)

    def domain_shape_compute(self):
        return (self.nic, self.njc, self.npz)

    def domain_shape_compute_buffer_2d(self):
        return (self.nic + 1, self.njc + 1, self.npz)
    
    def domain_shape_compute_x(self):
        return (self.nic + 1, self.njc, self.npz)
    
    def domain_shape_compute_y(self):
        return (self.nic, self.njc + 1, self.npz)

    def domain_x_compute_y(self):
        return (self.nid, self.njc, self.npz)

    def domain_y_compute_x(self):
        return (self.nic, self.njd, self.npz)
    
    def domain_shape_buffer_1cell(self):
        return (int(self.nid + 1), int(self.njd + 1), int(self.npz + 1))

    def domain_shape_y(self):
        return (int(self.nid), int(self.njd + 1), int(self.npz))

    def domain_shape_x(self):
        return (int(self.nid + 1), int(self.njd), int(self.npz))

    def corner_domain(self):
        return (1, 1, self.npz)

    def domain_shape_buffer_2d(self):
        return (int(self.nid + 1), int(self.njd + 1), int(self.npz))

    def copy_right_edge(self, var, i_index, j_index):
        return np.copy(var.data[i_index:, :, :]), np.copy(var.data[:, j_index:, :])

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var.data[:i_index, :, :] = edge_data_i
        var.data[:, :j_index, :] = edge_data_j
        
    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var.data[i_index:, :, :] = edge_data_i
        var.data[:, j_index:, :] = edge_data_j
    
    def uvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 2, self.je + 1)

    def vvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 1, self.je + 2)

    def edge_offset_halos(self, uvar, vvar):
        u_edge_i, u_edge_j = self.uvar_edge_halo(uvar)
        v_edge_i, v_edge_j = self.vvar_edge_halo(vvar)
        return u_edge_i, u_edge_j,  v_edge_i, v_edge_j
        
    def insert_edge(self, var, edge_data, index):
        var.data[index] = edge_data

    def append_edges(self, uvar, u_edge_i, u_edge_j, vvar, v_edge_i, v_edge_j):
        self.insert_right_edge(uvar, u_edge_i, self.ie+2, u_edge_j, self.je+1)
        self.insert_right_edge(vvar, v_edge_i, self.ie+1, v_edge_j, self.je+2)

    def overwrite_edges(self, var, edgevar, left_i_index, left_j_index):
        self.insert_left_edge(var, edgevar.data[:left_i_index, :, :], left_i_index, edgevar.data[:, :left_j_index, :], left_j_index)
        right_i_index = self.ie + left_i_index
        right_j_index = self.ie + left_j_index
        self.insert_right_edge(var, edgevar.data[right_i_index:, :, :], right_i_index, edgevar.data[:, right_j_index:, :], right_j_index)
        
    def compute_origin(self):
        return (self.is_, self.js, 0)

    def default_origin(self):
        return (self.isd, self.jsd, 0)

    def compute_x_origin(self):
        return (self.is_, self.jsd, 0)
    
    def compute_y_origin(self):
        return (self.isd, self.js, 0)
    
    # TODO, expand to more cases
    def horizontal_starts_from_shape(self, shape):
        if shape[0:2] in [self.domain_shape_compute()[0:2],
                          self.domain_shape_compute_x()[0:2],
                          self.domain_shape_compute_y()[0:2],
                          self.domain_shape_compute_buffer_2d()[0:2],
        ]:
            return self.is_, self.js
        else:
            return 0, 0

    def slice_data_k(self, ki):
        utils.k_slice_inplace(self.data_fields, ki)
        # update instance vars
        for k, v in self.data_fields.items():
            setattr(self, k, self.data_fields[k])
