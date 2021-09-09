from math import sin
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

def calculate_cos_sin_sg(xyz_dgrid, xyz_agrid, ec1, ec2, tile_partitioner, rank np):
    """
    Calculates the cosine and sine of the corner and side angles at each of the following points:
    8---3---7
    |       |
    0   4   2
    |       |
    5---1---6
    """
    shape_a = xyz_agrid.shape
    cos_sg = np.zeros((shape_a[0], shape_a[1], 9))
    sin_sg = np.zeros((shape_a[0], shape_a[1], 9))

    cos_sg[:, :, 5] = spherical_cos(xyz_dgrid[:-1, :-1, 0], xyz_dgrid[1:, :-1, 0], xyz_dgrid[:-1, 1:, 0])
    cos_sg[:, :, 6] = -1 * spherical_cos(xyz_dgrid[1:, :-1, 0], xyz_dgrid[:-1, :-1, 0], xyz_dgrid[1:, 1:, 0])
    cos_sg[:, :, 7] = spherical_cos(xyz_dgrid[1:, 1:, 0], xyz_dgrid[1:, :-1, 0], xyz_dgrid[:-1, 1:, 0])
    cos_sg[:, :, 8] = -1 * spherical_cos(xyz_dgrid[:-1, 1:, 0], xyz_dgrid[:-1, :-1, 0], xyz_dgrid[1:, 1:, 0])

    midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, 0], xyz_dgrid[:-1, 1:, 0])
    cos_sg[:, :, 0] = spherical_cos(midpoint, xyz_agrid[:, :, 0], xyz_dgrid[:-1, 1:, 0])
    midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, 0], xyz_dgrid[1:, :-1, 0])
    cos_sg[:, :, 1] = spherical_cos(midpoint, xyz_dgrid[1:, :-1, 0], xyz_agrid[:, :, 0])
    midpoint = xyz_midpoint(xyz_dgrid[1:, :-1, 0], xyz_dgrid[1:, 1:, 0])
    cos_sg[:, :, 2] = spherical_cos(midpoint, xyz_agrid[:, :, 0], xyz_dgrid[1:, :-1, 0])
    midpoint = xyz_midpoint(xyz_dgrid[:-1, 1:, 0], xyz_dgrid[1:, 1:, 0])
    cos_sg[:, :, 3] = spherical_cos(midpoint, xyz_dgrid[:-1, 1:, 0], xyz_agrid[:, :, 0])

    cos_sg[:, :, 4] = np.sum(ec1*ec2, axis=-1)

    sin_sg_tmp = 1.-cos_sg**2
    sin_sg_tmp[sin_sg_tmp < 0.] = 0.
    sin_sg = np.sqrt(sin_sg_tmp)
    sin_sg[sin_sg > 1.] = 1.

    #Adjust for corners:
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            pass #southwest corner
        elif tile_partitioner.on_tile_top(rank):
            pass #northwest corner
    elif tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            pass #southeast corner
        elif tile_partitioner.on_tile_top(rank):
            pass #northeast corner

    return cos_sg, sin_sg
