import functools # noqa: F401
from typing import Any, Dict

import fv3gfs.util as fv3util

from fv3core.grid import (
    get_area,
    gnomonic_grid,
    great_circle_distance_along_axis,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    mirror_grid,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
    init_grid,
    init_grid_utils,
)

from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid
from fv3core.utils.global_constants import LON_OR_LAT_DIM, PI, RADIUS, TILE_DIM
from fv3core.testing.parallel_translate import ParallelTranslateGrid


# TODO: After metric term code is all ported, could refactor code to use this container
# and prevent some of the back-and-forth conversion between lat/lon and x/y/z

# def metric_term(generating_function):
#     """Decorator which stores generated metric terms on `self` to be re-used in later
#     calls."""

#     @property
#     @functools.wraps(generating_function)
#     def wrapper(self):
#         hidden_name = '_' + generating_function.__name__
#         if not hasattr(self, hidden_name):
#             setattr(self, hidden_name, generating_function(self))
#         return getattr(self, hidden_name)
#     wrapper.metric_term = True
#     return wrapper


# class MetricTermContainer:

#     def __init__(self, **kwargs):
#         for name, value in **kwargs:
#             setattr(self, "_" + name, value)

#     def lon(self):
#         pass

#     def lat(self):
#         pass

#     def lon_agrid(self):
#         pass

#     def lat_agrid(self):
#         pass

#     @metric_term
#     def dx(self):
#         pass

#     @metric_term
#     def dy(self):
#         pass

#     @metric_term
#     def dx_agrid(self):
#         pass

#     @metric_term
#     def dy_agrid(self):
#         pass

#     @metric_term
#     def dx_cgrid(self):
#         pass

#     @metric_term
#     def dy_cgrid(self):
#         pass


class TranslateGnomonic_Grids(ParallelTranslateGrid):

    max_error = 2e-14

    inputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }
    outputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        for inputs in inputs_list:
            outputs.append(self.compute(inputs))
        return outputs

    def compute(self, inputs):
        state = self.state_from_inputs(inputs)
        gnomonic_grid(
            self.grid.grid_type,
            state["longitude_on_cell_corners"].view[:],
            state["latitude_on_cell_corners"].view[:],
            state["longitude_on_cell_corners"].np,
        )
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateMirror_Grid(ParallelTranslateGrid):

    inputs = {
        "grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
        "ng": {"name": "n_ghost", "dims": []},
        "npx": {"name": "npx", "dims": []},
        "npy": {"name": "npy", "dims": []},
    }
    outputs = {
        "grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        outputs.append(self.compute(inputs_list[0]))
        for inputs in inputs_list[1:]:
            outputs.append(inputs)
        return outputs

    def compute(self, inputs):
        state = self.state_from_inputs(inputs)
        mirror_grid(
            state["grid_global"].data,
            state["n_ghost"],
            state["npx"],
            state["npy"],
            state["grid_global"].np,
        )
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateGrid_Areas(ParallelTranslateGrid):

    inputs = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
    }
    outputs = {
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["area"].data[3:-4, 3:-4] = get_area(
            state["grid"].data[3:-3, 3:-3, 0],
            state["grid"].data[3:-3, 3:-3, 1],
            RADIUS,
            state["grid"].np,
        )
        state["area_cgrid"].data[3:-3, 3:-3] = get_area(
            state["agrid"].data[2:-3, 2:-3, 0],
            state["agrid"].data[2:-3, 2:-3, 1],
            RADIUS,
            state["grid"].np,
        )
        set_corner_area_to_triangle_area(
            lon=state["agrid"].data[2:-3, 2:-3, 0],
            lat=state["agrid"].data[2:-3, 2:-3, 1],
            area=state["area_cgrid"].data[3:-3, 3:-3],
            radius=RADIUS,
            np=state["grid"].np,
        )
        return state


class TranslateGrid_MoreAreas(ParallelTranslateGrid):

    inputs = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "rarea_c": {
            "name": "rarea_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }
    outputs = {
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        xyz_dgrid = lon_lat_to_xyz(
            state["grid"].data[:, :, 0], state["grid"].data[:, :, 1], state["grid"].np
        )
        xyz_agrid = lon_lat_to_xyz(
            state["agrid"].data[:-1, :-1, 0],
            state["agrid"].data[:-1, :-1, 1],
            state["agrid"].np,
        )
        set_c_grid_tile_border_area(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["area_cgrid"].data[3:-3, 3:-3],
            communicator.tile.partitioner,
            communicator.tile.rank,
            state["grid"].np,
        )
        set_tile_border_dxc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dx_cgrid"].data[3:-3, 3:-4],
            communicator.tile.partitioner,
            communicator.tile.rank,
            state["grid"].np,
        )
        set_tile_border_dyc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dy_cgrid"].data[3:-4, 3:-3],
            communicator.tile.partitioner,
            communicator.tile.rank,
            state["grid"].np,
        )
        return state


class TranslateGrid_Grid(ParallelTranslateGrid):

    max_error = 1e-14
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",}
    }
    outputs = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        shift_fac = 18
        grid_global = self.grid.quantity_factory.zeros(
            [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "radians",
            dtype=float,
        )
        lon = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        lat = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        gnomonic_grid(
            self.grid.grid_type,
            lon.view[:],
            lat.view[:],
            lon.np,
        )
        grid_global.view[:, :, 0, 0] = lon.view[:]
        grid_global.view[:, :, 1, 0] = lat.view[:]
        mirror_grid(
            grid_global.data,
            self.grid.halo,
            self.grid.npx,
            self.grid.npy,
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
            grid = self.grid.quantity_factory.empty(
                dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
                units="radians",
            )
            grid.data[:] = grid_global.data[:, :, :, i]
            state_list.append({"grid": grid})
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["grid"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
        for state in state_list:
            fill_corners_2d(
                state["grid"].data[:, :, :], self.grid, gridtype="B", direction="x"
            )
        return self.outputs_list_from_state_list(state_list)

    def compute_parallel(self, inputs, communicator):
        raise NotImplementedError()

    def compute(self, inputs):
        state = self.state_from_inputs(inputs)
        pass
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateGrid_DxDy(ParallelTranslateGrid):

    inputs = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }
    outputs = {
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        # before the halo update, the Fortran calls a get_symmetry routine
        # missing get_symmetry call in fv_grid_tools.F90, dy is set based on dx on
        # the opposite grid face, as a result dy has errors
        # (and dx in its halos from dy)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_vector_halo_update(
                    state["dx"], state["dy"], n_points=self.grid.halo
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
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["dx"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        state["dy"] = self.grid.quantity_factory.zeros(
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
        return state


class TranslateGrid_Agrid(ParallelTranslateGrid):

    inputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }
    outputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local_part1(inputs))
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["agrid"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
        for i, state in enumerate(state_list):
            fill_corners_2d(
                state["agrid"].data[:, :, 0][:, :, None],
                self.grid,
                gridtype="A",
                direction="x",
            )
            fill_corners_2d(
                state["agrid"].data[:, :, 1][:, :, None],
                self.grid,
                gridtype="A",
                direction="y",
            )
            state_list[i] = self._compute_local_part2(state)
        return self.outputs_list_from_state_list(state_list)

    def _compute_local_part1(self, inputs):
        state = self.state_from_inputs(inputs)
        lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        agrid_lon, agrid_lat = lon_lat_corner_to_cell_center(lon, lat, state["grid"].np)
        state["agrid"].data[:-1, :-1, 0], state["agrid"].data[:-1, :-1, 1] = (
            agrid_lon,
            agrid_lat,
        )
        return state

    def _compute_local_part2(self, state):
        lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        lon_y_center, lat_y_center = lon_lat_midpoint(
            lon[:, :-1], lon[:, 1:], lat[:, :-1], lat[:, 1:], state["grid"].np
        )
        dx_agrid = great_circle_distance_along_axis(
            lon_y_center, lat_y_center, RADIUS, state["grid"].np, axis=0
        )
        lon_x_center, lat_x_center = lon_lat_midpoint(
            lon[:-1, :], lon[1:, :], lat[:-1, :], lat[1:, :], state["grid"].np
        )
        dy_agrid = great_circle_distance_along_axis(
            lon_x_center, lat_x_center, RADIUS, state["grid"].np, axis=1
        )
        fill_corners_agrid(
            dx_agrid[:, :, None], dy_agrid[:, :, None], self.grid, vector=False
        )
        lon_agrid, lat_agrid = (
            state["agrid"].data[:-1, :-1, 0],
            state["agrid"].data[:-1, :-1, 1],
        )
        dx_cgrid = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, state["grid"].np, axis=0
        )
        dy_cgrid = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, state["grid"].np, axis=1
        )
        outputs = self.allocate_output_state()
        for name in ("dx_agrid", "dy_agrid"):
            state[name] = outputs[name]
        state["dx_agrid"].data[:-1, :-1] = dx_agrid
        state["dy_agrid"].data[:-1, :-1] = dy_agrid

        # copying the second-to-last values to the last values is what the Fortran
        # code does, but is this correct/valid?
        # Maybe we want to change this to use halo updates?
        state["dx_cgrid"].data[1:-1, :-1] = dx_cgrid
        state["dx_cgrid"].data[0, :-1] = dx_cgrid[0, :]
        state["dx_cgrid"].data[-1, :-1] = dx_cgrid[-1, :]

        state["dy_cgrid"].data[:-1, 1:-1] = dy_cgrid
        state["dy_cgrid"].data[:-1, 0] = dy_cgrid[:, 0]
        state["dy_cgrid"].data[:-1, -1] = dy_cgrid[:, -1]

        return state


class TranslateInitGrid(ParallelTranslateGrid):

    """need to add npx, npy, npz, ng
    !$ser
    data
    grid_file=grid_file
    ndims=ndims
    nregions=ntiles
    grid_name=grid_name
    sw_corner=Atm(n)%gridstruct%sw_corner
    se_corner=Atm(n)%gridstruct%se_corner
    ne_corner=Atm(n)%gridstruct%ne_corner
    nw_corner=Atm(n)%gridstruct%nw_corner
    """

    inputs = {
        "grid_file": {
            "name": "grid_spec_filename",
            "dims": [],
        },
        "ndims": {
            "name": "ndims",
            "dims": []
        },
        "nregions": {
            "name": "nregions",
            "dims": [],
        },
        "grid_name": {
            "name": "grid_name",
            "dims": [],
        },
        "sw_corner": {
            "name": "sw_corner",
            "dims": [],
        },
        "se_corner": {
            "name": "se_corner",
            "dims": [],
        },
        "nw_corner": {
            "name": "nw_corner",
            "dims": [],
        },
        "ne_corner": {
            "name": "ne_corner",
            "dims": [],
        }
    }
    """!$ser
data
iinta=Atm(n)%gridstruct%iinta
iintb=Atm(n)%gridstruct%iintb
jinta=Atm(n)%gridstruct%jinta
jintb=Atm(n)%gridstruct%jintb
gridvar=Atm(n)%gridstruct%grid_64
agrid=Atm(n)%gridstruct%agrid_64
area=Atm(n)%gridstruct%area_64
area_c=Atm(n)%gridstruct%area_c_64
rarea=Atm(n)%gridstruct%rarea
rarea_c=Atm(n)%gridstruct%rarea_c
dx=Atm(n)%gridstruct%dx_64
dy=Atm(n)%gridstruct%dy_64
dxc=Atm(n)%gridstruct%dxc_64
dyc=Atm(n)%gridstruct%dyc_64
dxa=Atm(n)%gridstruct%dxa_64
dya=Atm(n)%gridstruct%dya_64
rdx=Atm(n)%gridstruct%rdx
rdy=Atm(n)%gridstruct%rdy
rdxc=Atm(n)%gridstruct%rdxc
rdyc=Atm(n)%gridstruct%rdyc
rdxa=Atm(n)%gridstruct%rdxa
rdya=Atm(n)%gridstruct%rdya
latlon=Atm(n)%gridstruct%latlon
cubedsphere=Atm(n)%gridstruct%latlon
    """
    outputs: Dict[str, Any] = {
        "iinta": {
            "name": "i_int_a",
            "dims": [],
        },
        "iintb": {
            "name": "i_int_b",
            "dims": [],
        },
        "jinta": {
            "name": "j_int_a",
            "dims": [],
        },
        "jintb": {
            "name": "j_int_b",
            "dims": [],
        },
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "latlon": {
            "name": "latitude_longitude",
            "dims": [],
            "units": ""
        },
        "cubedsphere": {
            "name": "cubed_sphere",
            "dims": [],
            "units": "",
        }
    }

    def compute_sequential(self, inputs_list, communicator_list):
        outputs=[]
        for inputs in inputs_list:
            outputs.append(self._compute_local(inputs))
        return outputs

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        init_grid(state)
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateGridUtils_Init(ParallelTranslateGrid):

    """!$ser
    data
    gridvar=Atm(n)%gridstruct%grid_64
    agrid=Atm(n)%gridstruct%agrid_64
    area=Atm(n)%gridstruct%area_64
    area_c=Atm(n)%gridstruct%area_c_64
    rarea=Atm(n)%gridstruct%rarea
    rarea_c=Atm(n)%gridstruct%rarea_c
    dx=Atm(n)%gridstruct%dx_64
    dy=Atm(n)%gridstruct%dy_64
    dxc=Atm(n)%gridstruct%dxc_64
    dyc=Atm(n)%gridstruct%dyc_64
    dxa=Atm(n)%gridstruct%dxa_64
    dya=Atm(n)%gridstruct%dya_64"""

    inputs: Dict[str, Any] = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }
    """!$ser
data
edge_s=Atm(n)%gridstruct%edge_s
edge_n=Atm(n)%gridstruct%edge_n
edge_w=Atm(n)%gridstruct%edge_w
edge_e=Atm(n)%gridstruct%edge_e
del6_u=Atm(n)%gridstruct%del6_u
del6_v=Atm(n)%gridstruct%del6_v
divg_u=Atm(n)%gridstruct%divg_u
divg_v=Atm(n)%gridstruct%divg_v
cosa_u=Atm(n)%gridstruct%cosa_u
cosa_v=Atm(n)%gridstruct%cosa_v
cosa_s=Atm(n)%gridstruct%cosa_s
cosa=Atm(n)%gridstruct%cosa
sina_u=Atm(n)%gridstruct%sina_u
sina_v=Atm(n)%gridstruct%sina_v
rsin_u=Atm(n)%gridstruct%rsin_u
rsin_v=Atm(n)%gridstruct%rsin_v
rsina=Atm(n)%gridstruct%rsina
rsin2=Atm(n)%gridstruct%rsin2
sina=Atm(n)%gridstruct%sina
sin_sg=Atm(n)%gridstruct%sin_sg
cos_sg=Atm(n)%gridstruct%cos_sg
ks=Atm(n)%ks
ptop=Atm(n)%ptop
ak=Atm(n)%ak
bk=Atm(n)%bk
a11=Atm(n)%gridstruct%a11
a12=Atm(n)%gridstruct%a12
a21=Atm(n)%gridstruct%a21
a22=Atm(n)%gridstruct%a22
da_min=Atm(n)%gridstruct%da_min
da_max=Atm(n)%gridstruct%da_max
da_min_c=Atm(n)%gridstruct%da_min_c
da_max_c=Atm(n)%gridstruct%da_max_c
sw_corner=Atm(n)%gridstruct%sw_corner
se_corner=Atm(n)%gridstruct%se_corner
ne_corner=Atm(n)%gridstruct%ne_corner
nw_corner=Atm(n)%gridstruct%nw_corner"""
    outputs: Dict[str, Any] = {
        "aaa": {
            "name": "bbb",
            "dims": [],
            "units": "ccc"
        },
        "edge_s": {
            "name": "edge_south",
            "dims": [],
            "units": ""
        },
        "edge_n": {
            "name": "edge_north",
            "dims": [],
            "units": ""
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [],
            "units": ""
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [],
            "units": "ccc"
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [],
            "units": ""
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [],
            "units": ""
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [],
            "units": ""
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [],
            "units": ""
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [],
            "units": ""
        },
        "sina": {
            "name": "sina",
            "dims": [],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [],
            "units": ""
        },
        "cos_sg": {
            "name": "cos_sg",
            "dims": [],
            "units": ""
        },
        "ks": {
            "name": "ks",
            "dims": [],
            "units": ""
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": ""
        },
        "ak": {
            "name": "ak",
            "dims": [],
            "units": ""
        },
        "bk": {
            "name": "bk",
            "dims": [],
            "units": ""
        },
        "a11": {
            "name": "a11",
            "dims": [],
            "units": ""
        },
        "a12": {
            "name": "a12",
            "dims": [],
            "units": ""
        },
        "a21": {
            "name": "a21",
            "dims": [],
            "units": ""
        },
        "a22": {
            "name": "a22",
            "dims": [],
            "units": ""
        },
        "da_min": {
            "name": "da_min",
            "dims": [],
            "units": ""
        },
        "da_max": {
            "name": "da_max",
            "dims": [],
            "units": ""
        },
        "da_min_c": {
            "name": "da_min_c",
            "dims": [],
            "units": ""
        },
        "da_max_c": {
            "name": "da_max_c",
            "dims": [],
            "units": ""
        },
        "sw_corner": {
            "name": "sw_corner",
            "dims": [],
            "units": ""
        },
        "se_corner": {
            "name": "se_corner",
            "dims": [],
            "units": ""
        },
        "nw_corner": {
            "name": "nw_corner",
            "dims": [],
            "units": ""
        },
        "ne_corner": {
            "name": "ne_corner",
            "dims": [],
            "units": ""
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        outputs=[]
        for inputs in inputs_list:
            outputs.append(self._compute_local(inputs))
        return outputs

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        init_grid_utils(state)
        outputs = self.outputs_from_state(state)
        return outputs
