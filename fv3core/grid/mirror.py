import math

from ..utils.global_constants import N_TILES, PI, RADIUS, RIGHT_HAND_GRID


__all__ = ["mirror_grid"]

# A tile global version of mirror_grid
# Closer to the Fortran code
def global_mirror_grid(grid_global, ng: int, npx: int, npy: int, np):
    # first fix base region
    nreg = 0
    for j in range(0, math.ceil(npy / 2)):
        for i in range(0, math.ceil(npx / 2)):
            x1 = 0.25 * (
                np.abs(grid_global[ng + i, ng + j, 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 0, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg])
            )
            grid_global[ng + i, ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + j, 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + j, 0, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + npy - (j + 1), 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg]
            )

            y1 = 0.25 * (
                np.abs(grid_global[ng + i, ng + j, 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 1, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg])
            )

            grid_global[ng + i, ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + j, 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + j, 1, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + npy - (j + 1), 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg]
            )

            # force dateline/greenwich-meridion consitency
            if npx % 2 != 0:
                # TODO: this seems to not make a difference
                if i == (npx - 1) // 2:
                    grid_global[ng + i, ng + j, 0, nreg] = 0.0
                    grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = 0.0

    i_mid = (npx - 1) // 2
    j_mid = (npy - 1) // 2
    for nreg in range(1, N_TILES):
        for j in range(0, npy):
            x1 = grid_global[ng : ng + npx, ng + j, 0, 0]
            y1 = grid_global[ng : ng + npx, ng + j, 1, 0]
            z1 = RADIUS + 0.0 * x1

            if nreg == 1:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
            elif nreg == 2:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force North Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == i_mid:
                        x2[i_mid] = 0.0
                        y2[i_mid] = PI / 2.0
                    if j == j_mid:
                        x2[:i_mid] = 0.0
                        x2[i_mid + 1] = PI
            elif nreg == 3:
                ang = -180.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == (npy - 1) // 2:
                        x2[:] = PI
            elif nreg == 4:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
            elif nreg == 5:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 0.0
                x2, y2, z2 = _rot_3d(
                    3, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force South Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == i_mid:
                        x2[i_mid] = 0.0
                        y2[i_mid] = -PI / 2.0
                    if j > j_mid:
                        x2[i_mid] = 0.0
                    elif j < j_mid:
                        x2[i_mid] = PI

            grid_global[ng : ng + npx, ng + j, 0, nreg] = x2
            grid_global[ng : ng + npx, ng + j, 1, nreg] = y2

    return grid_global

def mirror_grid(grid_section, grid_mirror_ew, grid_mirror_ns, grid_mirror_diag, grid, tile_index, np):
    ng = grid.halo
    npx = grid.npx
    npy = grid.npy
    nreg = tile_index
  
    # first fix base region
    x_center_tile = np.all(grid_section == grid_mirror_ew)
    y_center_tile = np.all(grid_section == grid_mirror_ns)
    for j in range(grid.js, grid.je+2):
        for i in range(grid.is_, grid.ie+2):
           
            # NOTE brute force way to make sure you alays have the same order of operations to compute y1 from different ranks,
            # so we aren't chasing error ghosts as much as we might otherwise. 
            # shouldn't technically need this (the sum of 4 numbers is... the sum of 4 numbers), or could be way more clever about it
            if grid.global_is + i - ng <  ng + npx / 2:
                if grid.global_js + j - ng < ng + npy / 2:
                    if grid.rank == 0 and i == 4 and j == 13:
                        print('branch sw')
                    ll = grid_section[i, j, :]
                    lr = grid_mirror_ew[grid.ied+1 - i, j, :]
                    ul = grid_mirror_ns[i, grid.jed+1 - j, :]
                    ur = grid_mirror_diag[grid.ied+1 - i, grid.jed+1 - j, :]
                else:
                    if grid.rank == 0 and i == 4 and j == 13:
                        print('branch nw')
                    ll = grid_mirror_ns[i, grid.jed+1 - j, :]
                    lr = grid_mirror_diag[grid.ied+1 - i, grid.jed+1 - j, :]
                    ul = grid_section[i, j, :]
                    ur = grid_mirror_ew[grid.ied+1 - i, j, :]
            else:
                if grid.global_js + j - ng < ng + npy / 2:
                    if grid.rank == 0 and i == 4 and j == 13:
                        print('branch se')
                    ll = grid_mirror_ew[grid.ied+1 - i, j, :]
                    lr = grid_section[i, j, :]
                    ul = grid_mirror_diag[grid.ied+1 - i, grid.jed+1 - j, :]
                    ur = grid_mirror_ns[i, grid.jed+1 - j, :]
                else:
                    if grid.rank == 0 and i == 4 and j == 13:
                        print('branch ne')
                    ll = grid_mirror_diag[grid.ied+1 - i, grid.jed+1 - j, :]
                    lr = grid_mirror_ns[i, grid.jed+1 - j, :]
                    ul = grid_mirror_ew[grid.ied+1 - i, j, :]
                    ur = grid_section[i, j, :]
            # TODO, we can do this, just not for now tiny error
            x1 = 0.25 * (
                np.abs(grid_section[i, j, 0])
                + np.abs(grid_mirror_ew[grid.ied + 1- i, j, 0])
                + np.abs(grid_mirror_ns[i, grid.jed + 1 - j, 0])
                + np.abs(grid_mirror_diag[grid.ied +1 - i, grid.jed+1 - j, 0])
            )
            #x1 = 0.25 * (abs(ll[0]) + abs(lr[0]) + abs(ul[0]) + abs(ur[0]))
            grid_section[i, j, 0] = np.copysign(
                x1, grid_section[i, j, 0]
            )
           
            #y1 = 0.25 * (abs(ll[1]) + abs(lr[1]) + abs(ul[1]) + abs(ur[1]))
            
            y1 = 0.25 * (
                np.abs(grid_section[i, j, 1])
                + np.abs(grid_mirror_ew[grid.ied+1 - i, j, 1])
                + np.abs(grid_mirror_ns[i, grid.jed+1 - j, 1])
                + np.abs(grid_mirror_diag[grid.ied+1 - i, grid.jed+1 - j, 1])
            )
            
        
            grid_section[i, j, 1] = np.copysign(
                y1, grid_section[i, j, 1]
            )

            
            # force dateline/greenwich-meridion consistency
            # TODO This seems to have no impact
            if npx % 2 != 0:
                if x_center_tile and i == grid.is_ + (grid.ie+1 - grid.is_) // 2:
                    #if i == (npx - 1) // 2:
                    grid_section[i, j, 0] = 0.0

                     
    i_mid = (grid.ie+1 - grid.is_) // 2 
    j_mid = (grid.je+1 - grid.js) // 2 
   
    if nreg > 0:
        
        for j in range(grid.js, grid.je+2):
            x1 = grid_section[grid.is_ : grid.ie+2, j, 0]
            y1 = grid_section[grid.is_ : grid.ie+2, j, 1]
            z1 = RADIUS + 0.0 * x1

            if nreg == 1:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
            elif nreg == 2:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
               
                # force North Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == ng + i_mid and x_center_tile and y_center_tile:
                        x2[i_mid] = 0.0
                        y2[i_mid] = PI / 2.0
                    if j == ng + j_mid and y_center_tile:
                        if x_center_tile:
                            x2[:i_mid] = 0.0
                            x2[i_mid + 1] = PI
                        elif grid.global_is + i_mid < ng + (npx - 1) / 2:
                            x2[:] = 0.0
                        
            elif nreg == 3:
                ang = -180.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == ng + j_mid and y_center_tile:
                        x2[:] = PI
            elif nreg == 4:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
            elif nreg == 5:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 0.0
                x2, y2, z2 = _rot_3d(
                    3, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force South Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == ng + i_mid and x_center_tile and y_center_tile:
                        x2[i_mid] = 0.0
                        y2[i_mid] = -PI / 2.0
                    if grid.global_js + j_mid > ng+(npy - 1) / 2 and x_center_tile:
                        x2[i_mid] = 0.0
                    elif grid.global_js + j_mid < ng+(npy - 1) / 2 and x_center_tile:
                        x2[i_mid] = PI

            grid_section[grid.is_ : grid.ie+2, j, 0] = x2
            grid_section[grid.is_ : grid.ie+2, j, 1] = y2

    
def _rot_3d(axis, p, angle, np, degrees=False, convert=False):

    if convert:
        p1 = _spherical_to_cartesian(p, np)
    else:
        p1 = p

    if degrees:
        angle = np.deg2rad(angle)

    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 1:
        x2 = p1[0]
        y2 = c * p1[1] + s * p1[2]
        z2 = -s * p1[1] + c * p1[2]
    elif axis == 2:
        x2 = c * p1[0] - s * p1[2]
        y2 = p1[1]
        z2 = s * p1[0] + c * p1[2]
    elif axis == 3:
        x2 = c * p1[0] + s * p1[1]
        y2 = -s * p1[0] + c * p1[1]
        z2 = p1[2]
    else:
        assert False, "axis must be in [1,2,3]"

    if convert:
        p2 = _cartesian_to_spherical([x2, y2, z2], np)
    else:
        p2 = [x2, y2, z2]

    return p2


def _spherical_to_cartesian(p, np):
    lon, lat, r = p
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    if RIGHT_HAND_GRID:
        z = r * np.sin(lat)
    else:
        z = -r * np.sin(lat)
    return [x, y, z]


def _cartesian_to_spherical(p, np):
    x, y, z = p
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.where(np.abs(x) + np.abs(y) < 1.0e-10, 0.0, np.arctan2(y, x))
    if RIGHT_HAND_GRID:
        lat = np.arcsin(z / r)
    else:
        lat = np.arccos(z / r) - PI / 2.0
    return [lon, lat, r]

def set_halo_nan(grid, ng: int, np):
    grid[:ng, :, :] = np.nan #west edge
    grid[:, :ng, :] = np.nan #south edge
    grid[-ng:, :, :] = np.nan #east edge
    grid[:, -ng:, :] = np.nan #north edge
    return grid
