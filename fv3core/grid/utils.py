import typing
from fv3core.utils.global_constants import PI
from .gnomonic import lon_lat_to_xyz, xyz_midpoint, normalize_xyz, spherical_cos
import numpy as np

def set_eta(km, ks, ptop, ak, bk):
    """
    Sets the hybrid pressure coordinate
    """    
    pass

def var_hi(km, ak, bk, ptop, ks, pint, stretch_fac):
    pass

def get_center_vector(xyz_gridpoints, nhalo, np):

    if False: #ifdef OLD_VECT
        vector1 = xyz_gridpoints[1:, :-1, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[:-1, 1:, :]
        vector2 = xyz_gridpoints[:-1, 1:, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[1:, :-1, :]
    else:
        center_points = xyz_midpoint(
            xyz_gridpoints[:-1, :-1, 0],
            xyz_gridpoints[1:, :-1, 0],
            xyz_gridpoints[:-1, 1:, 0],
            xyz_gridpoints[1:, 1:, 0])
        
        p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, 0], xyz_gridpoints[:-1, 1:, ])
        p2 = xyz_midpoint(xyz_gridpoints[1:, :-1, 0], xyz_gridpoints[1:, 1:, 0])
        p3 = np.cross(p1, p2)
        vector1 = normalize_xyz(np.cross( center_points, p3))

        p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, 0], xyz_gridpoints[1:, :-1, 0])
        p2 = xyz_midpoint(xyz_gridpoints[:-1, 1:, 0], xyz_gridpoints[1:, 1:, 0])
        p3 = np.cross(p1, p2)
        vector2 = normalize_xyz(np.cross( center_points, p3))

    #set halo corners to 0
    vector1[:nhalo, :nhalo, :] = 0.
    vector1[:nhalo, -nhalo:, :] = 0.
    vector1[-nhalo:, :nhalo, :] = 0.
    vector1[-nhalo:, -nhalo:, :] = 0.

    vector2[:nhalo, :nhalo, :] = 0.
    vector2[:nhalo, -nhalo:, :] = 0.
    vector2[-nhalo:, :nhalo, :] = 0.
    vector2[-nhalo:, -nhalo:, :] = 0.

    return vector1, vector2

def calc_ew(xyz_dgrid, xyz_agrid, nhalo, tile_partitioner, rank, np):
    pp = xyz_midpoint(xyz_dgrid[1:-1,:-1,:3], xyz_dgrid[1:-1, 1:, :3])
    if tile_partitioner.on_tile_left(rank):
        p2 = np.cross(pp, xyz_agrid[1:,:,:3])
    elif tile_partitioner.on_tile_right(rank):
        p2 = np.cross(pp, xyz_agrid[:-1,:,:3])
    else:
        p2 = np.cross(xyz_agrid[:-1,:,:3], xyz_agrid[1:,:,:3])
    
    ew1 = normalize_xyz(np.cross(p2, pp))
    
    p1 = np.cross(xyz_dgrid[1:-1, :-1, 0], xyz_dgrid[1:-1, 1:, 0])
    ew2 = normalize_xyz(np.cross(p1, pp))

    ew = np.stack((ew1, ew2), axis=-1)

    ew[:nhalo, :nhalo, :, :] = 0.
    ew[:nhalo, -nhalo:, :, :] = 0.
    ew[-nhalo:, :nhalo, :, :] = 0.
    ew[-nhalo:, -nhalo:, :, :] = 0.
    
    return ew

def calc_es(xyz_dgrid, xyz_agrid, nhalo, tile_partitioner, rank, np):
    pp = xyz_midpoint(xyz_dgrid[:-1, 1:-1, :3], xyz_dgrid[1:, 1:-1, :3])
    if tile_partitioner.on_tile_bottom(rank):
        p2 = np.cross(pp, xyz_agrid[:, 1:, :3])
    elif tile_partitioner.on_tile_top(rank):
        p2 = np.cross(pp, xyz_agrid[:, :-1, :3])
    else:
        p2 = np.cross(xyz_agrid[:,:-1,:3], xyz_agrid[:, 1:, :3])
    
    es2 = normalize_xyz(np.cross(p2, pp))
    
    p1 = np.cross(xyz_dgrid[:-1, 1:-1, 0], xyz_dgrid[1:, 1:-1, 0])
    es1 = normalize_xyz(np.cross(p1, pp))

    es = np.stack((es1, es2), axis=-1)

    es[:nhalo, :nhalo, :, :] = 0.
    es[:nhalo, -nhalo:, :, :] = 0.
    es[-nhalo:, :nhalo, :, :] = 0.
    es[-nhalo:, -nhalo:, :, :] = 0.
    
    return es