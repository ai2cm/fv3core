from fv3core.grid.utils import set_eta, get_center_vector
from .gnomonic import (
    get_area,
    gnomonic_grid,
    great_circle_distance_along_axis,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
)

from .mirror import mirror_grid, set_halo_nan
from .geometry import set_eta

import fv3gfs.util as fv3util
from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM

def init_grid_sequential(grid_config, communicator_list):
    '''
    Creates a full grid
    '''
    pass
    shift_fac = 18
    grid_global = grid_config.quantity_factory.zeros(
        [
            fv3util.X_INTERFACE_DIM,
            fv3util.Y_INTERFACE_DIM,
            LON_OR_LAT_DIM,
            TILE_DIM,
        ],
        "radians",
        dtype=float,
    )
    # print(grid_global.np.shape(grid_global.data))
    lon = grid_config.quantity_factory.zeros(
        [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
    )
    lat = grid_config.quantity_factory.zeros(
        [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
    )
    gnomonic_grid(
        grid_config.grid_type,
        lon.view[:],
        lat.view[:],
        lon.np,
    )
    grid_global.view[:, :, 0, 0] = lon.view[:]
    grid_global.view[:, :, 1, 0] = lat.view[:]
    mirror_grid(
        grid_global.data,
        grid_config.halo,
        grid_config.npx,
        grid_config.npy,
        grid_global.np,
    )
    # Shift the corner away from Japan
    # This will result in the corner close to east coast of China
    grid_global.view[:, :, 0, :] -= PI / shift_fac
    lon = grid_global.data[:, :, 0, :]
    lon[lon < 0] += 2 * PI
    grid_global.data[grid_global.np.abs(grid_global.data[:]) < 1e-10] = 0.0
    state_list = []
    for i, inputs in enumerate(inputs_list):
        grid = grid_config.quantity_factory.empty(
            dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            units="radians",
        )
        grid.data[:] = grid_global.data[:, :, :, i]
        state_list.append({"grid": grid})
    req_list = []
    for state, communicator in zip(state_list, communicator_list):
        req_list.append(
            communicator.start_halo_update(state["grid"], n_points=grid_config.halo)
        )
    for communicator, req in zip(communicator_list, req_list):
        req.wait()
    for state in state_list:
        fill_corners_2d(
            state["grid"].data[:, :, :], grid_config, gridtype="B", direction="x"
        )
        state["grid"].data[:, :, :] = set_halo_nan(state["grid"].data[:, :, :], grid_config.halo, grid_global.np)

        state["dx"] = grid_config.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        state["dy"] = grid_config.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        state["dx"].view[:, :] = great_circle_distance_along_axis(
            state["grid"].view[:, :, 0],
            state["grid"].view[:, :, 1],
            RADIUS,
            state["grid"].np,
            axis=0,
        )
        state["dy"].view[:, :] = great_circle_distance_along_axis(
            state["grid"].view[:, :, 0],
            state["grid"].view[:, :, 1],
            RADIUS,
            state["grid"].np,
            axis=1,
        )
    req_list = []
    for state, communicator in zip(state_list, communicator_list):
        req_list.append(
            communicator.start_vector_halo_update(
                state["dx"], state["dy"], n_points=grid_config.halo
            )
        )
    for communicator, req in zip(communicator_list, req_list):
        req.wait()
        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
    for state, grid in zip(state_list, self.rank_grids):
        state["dx"].data[state["dx"].data < 0] *= -1
        state["dy"].data[state["dy"].data < 0] *= -1
        fill_corners_dgrid(
            state["dx"].data[:, :, None],
            state["dy"].data[:, :, None],
            grid,
            vector=False,
        )
    

    pass

def init_grid_utils(state):
    #init ak, bk, eta for cold start
    set_eta(npz, ks, ptop, ak, bk)
    pass