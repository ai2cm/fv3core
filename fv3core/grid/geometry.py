from math import sin
import typing
from fv3core.utils.global_constants import PI
from .gnomonic import lon_lat_to_xyz, xyz_midpoint, normalize_xyz, spherical_cos, get_unit_vector_direction, lon_lat_midpoint, get_lonlat_vect, great_circle_distance_lon_lat

def get_center_vector(xyz_gridpoints, grid_type, nhalo, tile_partitioner, rank, np):
    '''
    Calculates the unit vector pointing to the center of each grid cell.
    vector1 comes from using the halfway points of the left and top cell edges, while
    vector2 comes from using the halfway points of the bottom and right cell edges
    '''
    big_number = 1.e8

    if grid_type < 3:
        if False: #ifdef OLD_VECT
            vector1 = xyz_gridpoints[1:, :-1, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[:-1, 1:, :]
            vector2 = xyz_gridpoints[:-1, 1:, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[1:, :-1, :]
        else:
            center_points = xyz_midpoint(
                xyz_gridpoints[:-1, :-1, :],
                xyz_gridpoints[1:, :-1, :],
                xyz_gridpoints[:-1, 1:, :],
                xyz_gridpoints[1:, 1:, :])
            
            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[:-1, 1:, :])
            p2 = xyz_midpoint(xyz_gridpoints[1:, :-1, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p2, p1)
            vector1 = normalize_xyz(np.cross( center_points, p3))

            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[1:, :-1, :])
            p2 = xyz_midpoint(xyz_gridpoints[:-1, 1:, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p2, p1)
            vector2 = normalize_xyz(np.cross( center_points, p3))
      
        #fill ghost on ec1 and ec2:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(vector1, big_number, nhalo, "sw")
                _fill_ghost(vector2, big_number, nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(vector1, big_number, nhalo, "nw")
                _fill_ghost(vector2, big_number, nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(vector1, big_number, nhalo, "se")
                _fill_ghost(vector2, big_number, nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(vector1, big_number, nhalo, "ne")
                _fill_ghost(vector2, big_number, nhalo, "ne")
    else:
        shape_dgrid = xyz_gridpoints.shape
        vector1 = np.zeros((shape_dgrid[0]-1, shape_dgrid[1]-1, 3))
        vector2 = np.zeros((shape_dgrid[0]-1, shape_dgrid[1]-1, 3))
        vector1[:,:,0] = 1
        vector2[:,:,1] = 1
    
    return vector1, vector2

def calc_unit_vector_west(xyz_dgrid, xyz_agrid, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cartesian unit vector pointing west from every grid cell.
    The first set of values is the horizontal component, the second is the vertical component as 
    defined by the cell edges -- in a non-spherical grid these will be x and y unit vectors.

    """
    ew1 = np.zeros((xyz_dgrid.shape[0], xyz_agrid.shape[1], 3))
    ew2 = np.zeros((xyz_dgrid.shape[0], xyz_agrid.shape[1], 3))
    if grid_type < 3:
      
        pp = xyz_midpoint(xyz_dgrid[1:-1,:-1,:3], xyz_dgrid[1:-1, 1:, :3])
       
        p2 = np.cross(xyz_agrid[:-1,:,:3], xyz_agrid[1:,:,:3])
        if tile_partitioner.on_tile_left(rank):
            p2[nhalo - 1] = np.cross(pp[nhalo - 1], xyz_agrid[nhalo,:,:3])
        if tile_partitioner.on_tile_right(rank):
            p2[-nhalo] = np.cross(xyz_agrid[-nhalo - 1,:,:3], pp[-nhalo])
        

        ew1[1:-1,:,:] = normalize_xyz(np.cross(p2, pp))
        p1 = np.cross(xyz_dgrid[1:-1, :-1, :], xyz_dgrid[1:-1, 1:, :])
        ew2[1:-1,:,:] = normalize_xyz(np.cross(p1, pp))

        # ew = np.stack((ew1, ew2), axis=-1)
        
        #fill ghost on ew:
        
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(ew1, 0., nhalo, "sw")
                _fill_ghost(ew2, 0., nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(ew1, 0., nhalo, "nw")
                _fill_ghost(ew2, 0., nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(ew1, 0., nhalo, "se")
                _fill_ghost(ew2, 0., nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(ew1, 0., nhalo, "ne")
                _fill_ghost(ew2, 0., nhalo, "ne")
     
    else:
        ew1[:,:,1] = 1.
        ew2[:,:,2] = 1.
        # ew = np.stack((ew1, ew2), axis=-1)
    
    return ew1[1:-1,:,:], ew2[1:-1,:,:]

def calc_unit_vector_south(xyz_dgrid, xyz_agrid, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cartesian unit vector pointing south from every grid cell.
    The first set of values is the horizontal component, the second is the vertical component as 
    defined by the cell edges -- in a non-spherical grid these will be x and y unit vectors.
    """
    es1 = np.zeros((xyz_agrid.shape[0], xyz_dgrid.shape[1], 3))
    es2 = np.zeros((xyz_agrid.shape[0], xyz_dgrid.shape[1], 3))
    if grid_type < 3:
 
        pp = xyz_midpoint(xyz_dgrid[:-1, 1:-1, :3], xyz_dgrid[1:, 1:-1, :3])
        p2 = np.cross(xyz_agrid[:,:-1,:3], xyz_agrid[:, 1:, :3])
        if tile_partitioner.on_tile_bottom(rank):
            p2[:,nhalo - 1] = np.cross(pp[:,nhalo - 1], xyz_agrid[:, nhalo, :3])
        if tile_partitioner.on_tile_top(rank):
            p2[:,-nhalo] = np.cross(xyz_agrid[:, -nhalo - 1, :3], pp[:,-nhalo])
        
        es2[:, 1:-1,:] = normalize_xyz(np.cross(p2, pp))
        
        p1 = np.cross(xyz_dgrid[:-1, 1:-1, :], xyz_dgrid[1:, 1:-1, :])
        es1[:, 1:-1,:] = normalize_xyz(np.cross(p1, pp))

        # es = np.stack((es1, es2), axis=-1)
       
        #fill ghost on es:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(es1, 0., nhalo, "sw")
                _fill_ghost(es2, 0., nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(es1, 0., nhalo, "nw")
                _fill_ghost(es2, 0., nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(es1, 0., nhalo, "se")
                _fill_ghost(es2, 0., nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(es1, 0., nhalo, "ne")
                _fill_ghost(es2, 0., nhalo, "ne")
    else:
        es1[:,:,1] = 1.
        es2[:,:,2] = 1.
        # es = np.stack((es1, es2), axis=-1)
    
    return es1[:, 1:-1,:], es2[:, 1:-1,:]

def calculate_supergrid_cos_sin(xyz_dgrid, xyz_agrid, ec1, ec2, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cosine and sine of the grid angles at each of the following points in a supergrid cell:
    9---4---8
    |       |
    1   5   3
    |       |
    6---2---7
    """
    big_number = 1.e8
    tiny_number = 1.e-8

    shape_a = xyz_agrid.shape
    cos_sg = np.zeros((shape_a[0], shape_a[1], 9))+big_number
    sin_sg = np.zeros((shape_a[0], shape_a[1], 9))+tiny_number

    if grid_type < 3:
        cos_sg[:, :, 5] = spherical_cos(xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np)
        cos_sg[:, :, 6] = -1 * spherical_cos(xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np)
        cos_sg[:, :, 7] = spherical_cos(xyz_dgrid[1:, 1:, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np)
        cos_sg[:, :, 8] = -1 * spherical_cos(xyz_dgrid[:-1, 1:, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np)

        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[:-1, 1:, :])
        cos_sg[:, :, 0] = spherical_cos(midpoint, xyz_agrid[:, :, :], xyz_dgrid[:-1, 1:, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :])
        cos_sg[:, :, 1] = spherical_cos(midpoint, xyz_dgrid[1:, :-1, :], xyz_agrid[:, :, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[1:, :-1, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 2] = spherical_cos(midpoint, xyz_agrid[:, :, :], xyz_dgrid[1:, :-1, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[:-1, 1:, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 3] = spherical_cos(midpoint, xyz_dgrid[:-1, 1:, :], xyz_agrid[:, :, :], np)

        cos_sg[:, :, 4] = np.sum(ec1*ec2, axis=-1)

        sin_sg_tmp = 1.-cos_sg**2
        sin_sg_tmp[sin_sg_tmp < 0.] = 0.
        sin_sg = np.sqrt(sin_sg_tmp)
        sin_sg[sin_sg > 1.] = 1.

        #Adjust for corners:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank): #southwest corner
                sin_sg[nhalo-1,:nhalo,2] = sin_sg[:nhalo, nhalo, 1]
                sin_sg[:nhalo, nhalo-1, 3] = sin_sg[nhalo, :nhalo, 0]
            if tile_partitioner.on_tile_top(rank): #northwest corner
                sin_sg[nhalo -1, -nhalo:, 2] = sin_sg[:nhalo, -nhalo-1, 3][::-1]
                sin_sg[:nhalo, -nhalo, 1] = sin_sg[nhalo, -nhalo-2:-nhalo+1, 0][::-1]
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank): #southeast corner
                sin_sg[-nhalo, :nhalo, 0]  = sin_sg[-nhalo:, nhalo, 1][::-1]
                sin_sg[-nhalo:, nhalo-1, 3] = sin_sg[-nhalo-1, :nhalo, 2][::-1]
            if tile_partitioner.on_tile_top(rank): #northeast corner
                sin_sg[-nhalo, -nhalo:, 0] = sin_sg[-nhalo:, -nhalo-1, 3]
                sin_sg[-nhalo:, -nhalo, 1] = sin_sg[-nhalo-1, -nhalo:, 2]

    else:
        cos_sg[:] = 0.
        sin_sg[:] = 1.

    return cos_sg, sin_sg

def calculate_l2c_vu(dgrid, nhalo, np):
    #AAM correction

    point1v = dgrid[nhalo:-nhalo, nhalo:-nhalo-1, :]
    point2v = dgrid[nhalo:-nhalo, nhalo+1:-nhalo, :]
    midpoint_y = np.array(lon_lat_midpoint(
        point1v[:, :, 0], point2v[:, :, 0], 
        point1v[:, :, 1], point2v[:, :, 1], np
    )).transpose([1,2,0])
    unit_dir_y = get_unit_vector_direction(point1v, point2v, np)
    exv, eyv = get_lonlat_vect(midpoint_y, np)
    l2c_v = np.cos(midpoint_y[:,:,1]) * np.sum(unit_dir_y * exv, axis=-1)

    point1u = dgrid[nhalo:-nhalo-1, nhalo:-nhalo, :]
    point2u = dgrid[nhalo+1:-nhalo, nhalo:-nhalo, :]
    midpoint_x = np.array(lon_lat_midpoint(
        point1u[:, :, 0], point2u[:, :, 0], 
        point1u[:, :, 1], point2u[:, :, 1], np
    )).transpose([1,2,0])
    unit_dir_x = get_unit_vector_direction(point1u, point2u, np)
    exu, eyu = get_lonlat_vect(midpoint_x, np)
    l2c_u = np.cos(midpoint_x[:,:,1]) * np.sum(unit_dir_x * exu, axis=-1)

    return l2c_v, l2c_u

def generate_xy_unit_vectors(xyz_dgrid, nhalo, tile_partitioner, rank, np):
    cross_vect_x = np.cross(xyz_dgrid[nhalo-1:-nhalo-1, nhalo:-nhalo, :], xyz_dgrid[nhalo+1:-nhalo+1, nhalo:-nhalo, :])
    # print(cross_vect_x.shape)
    if tile_partitioner.on_tile_left(rank):
        cross_vect_x[0,:] = np.cross(xyz_dgrid[nhalo, nhalo:-nhalo, :], xyz_dgrid[nhalo+1, nhalo:-nhalo, :])
    if tile_partitioner.on_tile_right(rank):
        cross_vect_x[-1, :] = np.cross(xyz_dgrid[-nhalo-2, nhalo:-nhalo, :], xyz_dgrid[-nhalo-1, nhalo:-nhalo, :])
    unit_x_vector = normalize_xyz(np.cross(cross_vect_x, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo]))

    cross_vect_y = np.cross(xyz_dgrid[nhalo:-nhalo, nhalo-1:-nhalo-1, :], xyz_dgrid[nhalo:-nhalo, nhalo+1:-nhalo+1, :])
    if tile_partitioner.on_tile_bottom(rank):
        cross_vect_y[:,0] = np.cross(xyz_dgrid[nhalo:-nhalo, nhalo, :], xyz_dgrid[nhalo:-nhalo, nhalo+1, :])
    if tile_partitioner.on_tile_top(rank):
        cross_vect_y[:, -1] = np.cross(xyz_dgrid[nhalo:-nhalo, -nhalo-2, :], xyz_dgrid[nhalo:-nhalo, -nhalo-1, :])
    unit_y_vector = normalize_xyz(np.cross(cross_vect_y, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo]))

    return unit_x_vector, unit_y_vector

def calculate_trig_uv(xyz_dgrid, cos_sg, sin_sg, nhalo, tile_partitioner, rank, np):
    '''
    Calculates more trig quantities
    '''

    big_number = 1.e8
    tiny_number = 1.e-8

    dgrid_shape_2d = xyz_dgrid[:,:,0].shape
    cosa = np.zeros(dgrid_shape_2d)+big_number
    sina = np.zeros(dgrid_shape_2d)+big_number
    cosa_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1]-1))+big_number
    sina_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1]-1))+big_number
    rsin_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1]-1))+big_number
    cosa_v = np.zeros((dgrid_shape_2d[0]-1, dgrid_shape_2d[1]))+big_number
    sina_v = np.zeros((dgrid_shape_2d[0]-1, dgrid_shape_2d[1]))+big_number
    rsin_v = np.zeros((dgrid_shape_2d[0]-1, dgrid_shape_2d[1]))+big_number

    cosa[nhalo:-nhalo, nhalo:-nhalo] = 0.5*(cos_sg[nhalo-1:-nhalo, nhalo-1:-nhalo, 7] + cos_sg[nhalo:-nhalo+1, nhalo:-nhalo+1, 5])
    sina[nhalo:-nhalo, nhalo:-nhalo] = 0.5*(sin_sg[nhalo-1:-nhalo, nhalo-1:-nhalo, 7] + sin_sg[nhalo:-nhalo+1, nhalo:-nhalo+1, 5])

    cosa_u[1:-1,:] = 0.5*(cos_sg[:-1,:,2] + cos_sg[1:,:,0])
    sina_u[1:-1,:] = 0.5*(sin_sg[:-1,:,2] + sin_sg[1:,:,0])
    sinu2 = sina_u[1:-1,:]**2
    sinu2[sinu2 < tiny_number] = tiny_number
    rsin_u[1:-1,:] = 1./sinu2

    cosa_v[:,1:-1] = 0.5*(cos_sg[:,:-1,3] + cos_sg[:,1:,1])
    sina_v[:,1:-1] = 0.5*(sin_sg[:,:-1,3] + sin_sg[:,1:,1])
    sinv2 = sina_v[:,1:-1]**2
    sinv2[sinv2 < tiny_number] = tiny_number
    rsin_v[:,1:-1] = 1./sinv2

    cosa_s = cos_sg[:,:,4]
    sin2 = sin_sg[:,:,4]**2
    sin2[sin2 < tiny_number] = tiny_number
    rsin2 = 1./sin2

    #fill ghost on cosa_s:
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "sw")
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "nw")
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "se")
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "ne")

    sina2 = sina[nhalo:-nhalo, nhalo:-nhalo]**2
    sina2[sina2 < tiny_number] = tiny_number
    rsina = 1./sina2

    # Set special sin values at edges
    if tile_partitioner.on_tile_left(rank):
        rsina[0, :] = big_number
        sina_u_limit = sina_u[nhalo,:]
        sina_u_limit[abs(sina_u_limit) < tiny_number] = tiny_number * np.sign(sina_u_limit[abs(sina_u_limit) < tiny_number])
        rsin_u[nhalo,:] = 1./sina_u_limit
    if tile_partitioner.on_tile_right(rank):
        rsina[-1, :] = big_number
        sina_u_limit = sina_u[-nhalo-1,:]
        sina_u_limit[abs(sina_u_limit) < tiny_number] = tiny_number * np.sign(sina_u_limit[abs(sina_u_limit) < tiny_number])
        rsin_u[-nhalo-1,:] = 1./sina_u_limit
    if tile_partitioner.on_tile_bottom(rank):
        rsina[:, 0] = big_number
        sina_v_limit = sina_v[:,nhalo]
        sina_v_limit[abs(sina_v_limit) < tiny_number] = tiny_number * np.sign(sina_v_limit[abs(sina_v_limit) < tiny_number])
        rsin_v[:,nhalo] = 1./sina_v_limit
    if tile_partitioner.on_tile_top(rank):
        rsina[:,-1] = big_number
        sina_v_limit = sina_v[:,-nhalo-1]
        sina_v_limit[abs(sina_v_limit) < tiny_number] = tiny_number * np.sign(sina_v_limit[abs(sina_v_limit) < tiny_number])
        rsin_v[:,-nhalo-1] = 1./sina_v_limit

    return cosa, sina, cosa_u, cosa_v, cosa_s, sina_u, sina_v, rsin_u, rsin_v, rsina, rsin2

def supergrid_corner_fix(cos_sg, sin_sg, nhalo, tile_partitioner, rank):
    """
    _fill_ghost overwrites some of the sin_sg 
    values along the outward-facing edge of a tile in the corners, which is incorrect. 
    This function resolves the issue by filling in the appropriate values after the _fill_ghost call
    """
    big_number = 1.e8
    tiny_number = 1.e-8
    
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "sw")
            _fill_ghost(cos_sg, big_number, nhalo, "sw")
            _rotate_trig_sg_sw_counterclockwise(sin_sg[:,:,1], sin_sg[:,:,2], nhalo)
            _rotate_trig_sg_sw_counterclockwise(cos_sg[:,:,1], cos_sg[:,:,2], nhalo)
            _rotate_trig_sg_sw_clockwise(sin_sg[:,:,0], sin_sg[:,:,3], nhalo)
            _rotate_trig_sg_sw_clockwise(cos_sg[:,:,0], cos_sg[:,:,3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "nw")
            _fill_ghost(cos_sg, big_number, nhalo, "nw")
            _rotate_trig_sg_nw_counterclockwise(sin_sg[:,:,0], sin_sg[:,:,1], nhalo)
            _rotate_trig_sg_nw_counterclockwise(cos_sg[:,:,0], cos_sg[:,:,1], nhalo)
            _rotate_trig_sg_nw_clockwise(sin_sg[:,:,3], sin_sg[:,:,2], nhalo)
            _rotate_trig_sg_nw_clockwise(cos_sg[:,:,3], cos_sg[:,:,2], nhalo)
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "se")
            _fill_ghost(cos_sg, big_number, nhalo, "se")
            _rotate_trig_sg_se_clockwise(sin_sg[:,:,1], sin_sg[:,:,0], nhalo)
            _rotate_trig_sg_se_clockwise(cos_sg[:,:,1], cos_sg[:,:,0], nhalo)
            _rotate_trig_sg_se_counterclockwise(sin_sg[:,:,2], sin_sg[:,:,3], nhalo)
            _rotate_trig_sg_se_counterclockwise(cos_sg[:,:,2], cos_sg[:,:,3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "ne")
            _fill_ghost(cos_sg, big_number, nhalo, "ne")
            _rotate_trig_sg_ne_counterclockwise(sin_sg[:,:,3], sin_sg[:,:,0], nhalo)
            _rotate_trig_sg_ne_counterclockwise(cos_sg[:,:,3], cos_sg[:,:,0], nhalo)
            _rotate_trig_sg_ne_clockwise(sin_sg[:,:,2], sin_sg[:,:,1], nhalo)
            _rotate_trig_sg_ne_clockwise(cos_sg[:,:,2], cos_sg[:,:,1], nhalo)


def _rotate_trig_sg_sw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    sg_field_out[nhalo-1, :nhalo] = sg_field_in[:nhalo,nhalo]

def _rotate_trig_sg_sw_clockwise(sg_field_in, sg_field_out, nhalo):    
    sg_field_out[:nhalo, nhalo-1] = sg_field_in[nhalo, :nhalo]

def _rotate_trig_sg_nw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[:,::-1], sg_field_out[:,::-1], nhalo)

def _rotate_trig_sg_nw_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[:,::-1], sg_field_out[:,::-1], nhalo)

def _rotate_trig_sg_se_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[::-1,:], sg_field_out[::-1,:], nhalo)

def _rotate_trig_sg_se_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[::-1,:], sg_field_out[::-1,:], nhalo)

def _rotate_trig_sg_ne_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[::-1,::-1], sg_field_out[::-1,::-1], nhalo)

def _rotate_trig_sg_ne_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[::-1,::-1], sg_field_out[::-1,::-1], nhalo)


def calculate_divg_del6(sin_sg, sina_u, sina_v, dx, dy, dxc, dyc, nhalo, tile_partitioner, rank):
    
    divg_u = sina_v * dyc / dx
    del6_u = sina_v * dx / dyc
    divg_v = sina_u * dxc / dy
    del6_v = sina_u * dy / dxc

    if tile_partitioner.on_tile_bottom(rank):
        divg_u[:, nhalo] = 0.5*(sin_sg[:, nhalo, 1] + sin_sg[:, nhalo-1, 3])*dyc[:, nhalo] / dx[:, nhalo]
        del6_u[:, nhalo] = 0.5*(sin_sg[:, nhalo, 1] + sin_sg[:, nhalo-1, 3])*dx[:, nhalo] / dyc[:, nhalo]
    if tile_partitioner.on_tile_top(rank):
        divg_u[:, -nhalo-1] = 0.5*(sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo-1, 3])*dyc[:, -nhalo-1] / dx[:, -nhalo-1]
        del6_u[:, -nhalo-1] = 0.5*(sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo-1, 3])*dx[:, -nhalo-1] / dyc[:, -nhalo-1]
    if tile_partitioner.on_tile_left(rank):
        divg_v[nhalo, :] = 0.5*(sin_sg[nhalo, :, 0] + sin_sg[nhalo-1, :, 2])*dxc[nhalo, :] / dy[nhalo, :]
        del6_v[nhalo, :] = 0.5*(sin_sg[nhalo, :, 0] + sin_sg[nhalo-1, :, 2])*dy[nhalo, :] / dxc[nhalo, :]
    if tile_partitioner.on_tile_right(rank):
        divg_v[-nhalo-1, :] = 0.5*(sin_sg[-nhalo, :, 0] + sin_sg[-nhalo-1, :, 2])*dxc[-nhalo-1, :] / dy[-nhalo-1, :]
        del6_v[-nhalo-1, :] = 0.5*(sin_sg[-nhalo, :, 0] + sin_sg[-nhalo-1, :, 2])*dy[-nhalo-1, :] / dxc[-nhalo-1, :]

    return divg_u, divg_v, del6_u, del6_v

def calculate_grid_z(ec1, ec2, vlon, vlat, np):
    z11 = np.sum(ec1 * vlon, axis=-1)
    z12 = np.sum(ec1 * vlat, axis=-1)
    z21 = np.sum(ec2 * vlon, axis=-1)
    z22 = np.sum(ec2 * vlat, axis=-1)
    return z11, z12, z21, z22

def calculate_grid_a(z11, z12, z21, z22,  sin_sg5):
    a11 = 0.5*z22/sin_sg5
    a12 = -0.5*z12/sin_sg5
    a21 = -0.5*z21/sin_sg5
    a22 = 0.5*z11/sin_sg5
    return a11, a12, a21, a22

def edge_factors(grid, agrid, grid_type, nhalo, tile_partitioner, rank, radius, np):
    """
    Creates interpolation factors from the A grid to the B grid on face edges
    """
    big_number = 1.e8
    npx = grid[nhalo:-nhalo, nhalo:-nhalo].shape[0]
    npy = grid[nhalo:-nhalo, nhalo:-nhalo].shape[1]
    edge_n = np.zeros(npx)+big_number
    edge_s = np.zeros(npx)+big_number
    edge_e = np.zeros(npy)+big_number
    edge_w = np.zeros(npy)+big_number

    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            edge_w[1:-1] = set_west_edge_factor(grid, agrid, nhalo, radius, np)
        if tile_partitioner.on_tile_right(rank):
            edge_e[1:-1] = set_east_edge_factor(grid, agrid, nhalo, radius, np)
        if tile_partitioner.on_tile_bottom(rank):
            edge_s[1:-1] = set_south_edge_factor(grid, agrid, nhalo, radius, np)
        if tile_partitioner.on_tile_bottom(rank):
            edge_n[1:-1] = set_north_edge_factor(grid, agrid, nhalo, radius, np)

    return edge_w, edge_e, edge_s, edge_n

def set_west_edge_factor(grid, agrid, nhalo, radius, np):
    py0, py1 = lon_lat_midpoint(agrid[nhalo-1, nhalo:-nhalo, 0], agrid[nhalo, nhalo:-nhalo, 0], agrid[nhalo-1, nhalo:-nhalo, 1], agrid[nhalo, nhalo:-nhalo, 1], np)
    d1 = great_circle_distance_lon_lat(py0[:-1], grid[nhalo,nhalo+1:-nhalo-1,0], py1[:-1], grid[nhalo,nhalo+1:-nhalo-1,1], radius, np)
    d2 = great_circle_distance_lon_lat(py0[1:], grid[nhalo,nhalo+1:-nhalo-1,0], py1[1:], grid[nhalo,nhalo+1:-nhalo-1,1], radius, np)
    west_edge_factor = d2/(d1+d2)
    return west_edge_factor

def set_east_edge_factor(grid, agrid, nhalo, radius, np):
    return set_west_edge_factor(grid[::-1, :, :], agrid[::-1, :, :], nhalo, radius, np)

def set_south_edge_factor(grid, agrid, nhalo, radius, np):
    return set_west_edge_factor(grid.transpose([1,0,2]), agrid.transpose([1,0,2]), nhalo, radius, np)

def set_north_edge_factor(grid, agrid, nhalo, radius, np):
    return set_west_edge_factor(grid[:, ::-1, :].transpose([1,0,2]), agrid[:, ::-1, :].transpose([1,0,2]), nhalo, radius, np)

def efactor_a2c_v(grid, agrid, grid_type, nhalo, tile_partitioner, rank, radius, np):
    '''
    Creates interpolation factors at face edges to interpolate from A to C grids
    '''
    big_number = 1.e8
    npx = grid.shape[0]-2*nhalo
    npy = grid.shape[1]-2*nhalo
    if npx != npy: raise ValueError("npx must equal npy")
    if npx %2 == 0: raise ValueError("npx must be odd")

    im2 = int((npx-1)/2)
    jm2 = int((npy-1)/2)

    d2 = d1 = np.zeros(npy+1)

    edge_vect_s = edge_vect_n = np.zeros(grid.shape[0]-1)+ big_number
    edge_vect_e = edge_vect_w = np.zeros(grid.shape[1]-1)+ big_number

    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            edge_vect_w[2:-2] = calculate_west_edge_vectors(grid, agrid, jm2, nhalo, radius, np)
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_w[nhalo-1] = edge_vect_w[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_w[-nhalo] = edge_vect_w[-nhalo-1]
        if tile_partitioner.on_tile_right(rank):
            edge_vect_e[2:-2] = calculate_east_edge_vectors(grid, agrid, jm2, nhalo, radius, np)
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_e[nhalo-1] = edge_vect_e[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_e[-nhalo] = edge_vect_e[-nhalo-1]
        if tile_partitioner.on_tile_bottom(rank):
            edge_vect_s[2:-2] = calculate_south_edge_vectors(grid, agrid, im2, nhalo, radius, np)
            if tile_partitioner.on_tile_left(rank):
                edge_vect_s[nhalo-1] = edge_vect_s[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_s[-nhalo] = edge_vect_s[-nhalo-1]
        if tile_partitioner.on_tile_top(rank):
            edge_vect_n[2:-2] = calculate_north_edge_vectors(grid, agrid, im2, nhalo, radius, np)
            if tile_partitioner.on_tile_left(rank):
                edge_vect_n[nhalo-1] = edge_vect_n[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_n[-nhalo] = edge_vect_n[-nhalo-1]

    return edge_vect_w, edge_vect_e, edge_vect_s, edge_vect_n
    
def calculate_west_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    d2 = d1 = np.zeros(grid.shape[0]-2*nhalo+1)
    py0, py1 = lon_lat_midpoint(agrid[nhalo-1, nhalo-2:-nhalo+2, 0], agrid[nhalo, nhalo-2:-nhalo+2, 0], agrid[nhalo-1, nhalo-2:-nhalo+2, 1], agrid[nhalo, nhalo-2:-nhalo+2, 1], np)
    p20, p21 = lon_lat_midpoint(grid[nhalo, nhalo-2:-nhalo+1, 0], grid[nhalo, nhalo-1:-nhalo+2, 0], grid[nhalo, nhalo-2:-nhalo+1, 1], grid[nhalo, nhalo-1:-nhalo+2, 1], np)
    py = np.array([py0, py1]).transpose([1,0])
    p2 = np.array([p20, p21]).transpose([1,0])
    d1[:jm2+1] = great_circle_distance_lon_lat(py[1:jm2+2, 0], p2[1:jm2+2, 0], py[1:jm2+2, 1], p2[1:jm2+2, 1], radius, np)
    d2[:jm2+1] = great_circle_distance_lon_lat(py[2:jm2+3, 0], p2[1:jm2+2, 0], py[2:jm2+3, 1], p2[1:jm2+2, 1], radius, np)
    d1[jm2+1:] = great_circle_distance_lon_lat(py[jm2+2:-1, 0], p2[jm2+2:-1, 0], py[jm2+2:-1, 1], p2[jm2+2:-1, 1], radius, np)
    d2[jm2+1:] = great_circle_distance_lon_lat(py[jm2+1:-2, 0], p2[jm2+2:-1, 0], py[jm2+1:-2, 1], p2[jm2+2:-1, 1], radius, np)
    return d1/(d2+d1)

def calculate_east_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    return calculate_west_edge_vectors(grid[::-1, :, :], agrid[::-1, :, :], jm2, nhalo, radius, np)

def calculate_south_edge_vectors(grid, agrid, im2, nhalo, radius, np):
    return calculate_west_edge_vectors(grid.transpose([1,0,2]), agrid.transpose([1,0,2]), im2, nhalo, radius, np)

def calculate_north_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    return calculate_west_edge_vectors(grid[:, ::-1, :].transpose([1,0,2]), agrid[:, ::-1, :].transpose([1,0,2]), jm2, nhalo, radius, np)

def unit_vector_lonlat(grid, np):
    '''
    Calculates the cartesian unit vectors for each point on a lat/lon grid
    '''

    sin_lon = np.sin(grid[:,:,0])
    cos_lon = np.cos(grid[:,:,0])
    sin_lat = np.sin(grid[:,:,1])
    cos_lat = np.cos(grid[:,:,1])

    unit_lon = np.array([-sin_lon, cos_lon, np.zeros(grid[:,:,0].shape)]).transpose([1,2,0])
    unit_lat = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]).transpose([1,2,0])

    return unit_lon, unit_lat

def _fill_ghost(field, value: float, nhalo: int, corner: str):
    """
    Fills a tile halo corner (ghost cells) of a field with a set value along the first 2 axes
    Args:
        field: the field to fill in, assumed to have x and y as the first 2 dimensions
        value: the value to fill
        nhalo: the number of halo points in the field
        corner: which corner to fill
    """
    if (corner == "sw") or (corner == "southwest"):
        field[:nhalo, :nhalo] = value
    elif (corner == "nw") or (corner == "northwest"):
        field[:nhalo, -nhalo:] = value
    elif (corner == "se") or (corner == "southeast"):
        field[-nhalo:, :nhalo] = value
    elif (corner == "ne") or (corner == "northeast"):
        field[-nhalo:, -nhalo:] = value
    else:
        raise ValueError("fill ghost requires a corner to be one of: sw, se, nw, ne")
    
