import functools
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
    init_grid_sequential,
    init_grid_utils,
    set_halo_nan,
)
from fv3core.grid.geometry import (
    get_center_vector,
    calc_unit_vector_west,
    calc_unit_vector_south,
    calculate_cos_sin_sg,
    calculate_l2c_vu,
    sg_corner_fix,
    calculate_divg_del6,
    init_cubed_to_latlon,
    unit_vector_lonlat,
    calculate_grid_z, 
    calculate_grid_a,
    edge_factors,
    efactor_a2c_v
)
from fv3core.grid.eta import set_eta

from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM
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


class TranslateGnomonicGrids(ParallelTranslateGrid):

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


class TranslateMirrorGrid(ParallelTranslateGrid):

    inputs = {
        "master_grid_global": {
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
        "master_ng": {"name": "n_ghost", "dims": []},
        "master_npx": {"name": "npx", "dims": []},
        "master_npy": {"name": "npy", "dims": []},
    }
    outputs = {
        "master_grid_global": {
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


class TranslateGridAreas(ParallelTranslateGrid):

    inputs = {
        "grid": {
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


class TranslateMoreAreas(ParallelTranslateGrid):

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
        "area_cgrid": {
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
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_vector_halo_update(
                    state["dx_cgrid"], state["dy_cgrid"], n_points=self.grid.halo
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
        for state, grid in zip(state_list, self.rank_grids):
            #TODO: Add support for unsigned vector halo updates instead of handling ad-hoc here
            state["dx_cgrid"].data[state["dx_cgrid"].data < 0] *= -1
            state["dy_cgrid"].data[state["dy_cgrid"].data < 0] *= -1

            #TODO: fix issue with interface dimensions causing validation errors
            fill_corners_cgrid(
                state["dx_cgrid"].data[:, :, None],
                state["dy_cgrid"].data[:, :, None],
                grid,
                vector=False,
            )

        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["area_cgrid"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()

        for i, state in enumerate(state_list):
            fill_corners_2d(
                state["area_cgrid"].data[:, :, None][:, :, None],
                self.grid,
                gridtype="B",
                direction="x",
            )
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
            xyz_dgrid[2:-2, 2:-2, :],
            xyz_agrid[2:-2, 2:-2, :],
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


class TranslateGridGrid(ParallelTranslateGrid):

    max_error = 1e-14
    inputs: Dict[str, Any] = {
        "grid_global": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM, TILE_DIM],
            "units": "radians",}
    }
    outputs = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }

    def __init__(self, grids):
        super().__init__(grids)
        self.max_error = 1.e-13

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
        # print(grid_global.np.shape(grid_global.data))
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
            # state["grid"].data[:, :, :] = set_halo_nan(state["grid"].data[:, :, :], self.grid.halo, grid_global.np)
        return self.outputs_list_from_state_list(state_list)

    def compute_parallel(self, inputs, communicator):
        raise NotImplementedError()

    # def compute(self, inputs):
    #     state = self.state_from_inputs(inputs)
    #     pass
    #     outputs = self.outputs_from_state(state)
    #     return outputs


class TranslateDxDy(ParallelTranslateGrid):

    inputs = {
        "grid": {
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


class TranslateAGrid(ParallelTranslateGrid):

    inputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "grid": {
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
        "grid": {
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
        lon_y_center, lat_y_center =  lon_lat_midpoint(
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

    """need to add npx, npy, npz, ng?
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

    def __init__(self, grids):
        super().__init__(grids)
        self.ignore_near_zero_errors = {}
        self.ignore_near_zero_errors["grid"] = True

    def compute_sequential(self, inputs_list, communicator_list):

        #Set up initial lat-lon d-grid
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
        # print(grid_global.np.shape(grid_global.data))
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
        for i, state in enumerate(state_list):
            fill_corners_2d(
                state["grid"].data[:, :, :], self.grid, gridtype="B", direction="x"
            )
            state_list[i] = state


        #calculate d-grid cell side lengths
        for i, state in enumerate(state_list):
            state_list[i] = self._compute_local_dxdy(state)
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
        

        #Set up lat-lon a-grid, calculate side lengths on a-grid
        for i, state in enumerate(state_list):
            state_list[i] = self._compute_local_agrid_part1(state)
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
            state_list[i] = self._compute_local_agrid_part2(state)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_vector_halo_update(
                    state["dx_agrid"], state["dy_agrid"], n_points=self.grid.halo
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
            # at this point the Fortran code copies in the west and east edges from
            # the halo for dy and performs a halo update,
            # to ensure dx and dy mirror across the boundary.
            # Not doing it here at the moment.
        for state, grid in zip(state_list, self.rank_grids):
            state["dx_agrid"].data[state["dx_agrid"].data < 0] *= -1
            state["dy_agrid"].data[state["dy_agrid"].data < 0] *= -1


        #Calculate a-grid areas and initial c-grid area
        for i, state in enumerate(state_list):
            state_list[i] = self._compute_local_areas_pt1(state)
            

        #Finish c-grid areas, calculate sidelengths on the c-grid
        for i, state in enumerate(state_list):
            state_list[i] = (self._compute_local_areas_pt2(state, communicator_list[i]))
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_vector_halo_update(
                    state["dx_cgrid"], state["dy_cgrid"], n_points=self.grid.halo
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
        for state, grid in zip(state_list, self.rank_grids):
            #TODO: Add support for unsigned vector halo updates instead of handling ad-hoc here
            state["dx_cgrid"].data[state["dx_cgrid"].data < 0] *= -1
            state["dy_cgrid"].data[state["dy_cgrid"].data < 0] *= -1

            #TODO: fix issue with interface dimensions causing validation errors
            fill_corners_cgrid(
                state["dx_cgrid"].data[:, :, None],
                state["dy_cgrid"].data[:, :, None],
                grid,
                vector=False,
            )

        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["area"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()

        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["area_cgrid"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()

        for i, state in enumerate(state_list):
            fill_corners_2d(
                state["area_cgrid"].data[:, :, None][:, :, None],
                self.grid,
                gridtype="B",
                direction="x",
            )
        return self.outputs_list_from_state_list(state_list)


    def _compute_local_dxdy(self, state):
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


    def _compute_local_agrid_part1(self, state):
        state["agrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM], "radians"
        )
        lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        agrid_lon, agrid_lat = lon_lat_corner_to_cell_center(lon, lat, state["grid"].np)
        state["agrid"].data[:-1, :-1, 0], state["agrid"].data[:-1, :-1, 1] = (
            agrid_lon,
            agrid_lat,
        )
        return state

    def _compute_local_agrid_part2(self, state):
        state["dx_agrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_agrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dx_cgrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_cgrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        lon_y_center, lat_y_center =  lon_lat_midpoint(
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


    def _compute_local_areas_pt1(self, state):
        state["area"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m^2"
        )
        state["area"].data[:, :] = -1.e8
        state["area_cgrid"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "m^2"
        )
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

    def _compute_local_areas_pt2(self, state, communicator):
        xyz_dgrid = lon_lat_to_xyz(
            state["grid"].data[:, :, 0], state["grid"].data[:, :, 1], state["grid"].np
        )
        xyz_agrid = lon_lat_to_xyz(
            state["agrid"].data[:-1, :-1, 0],
            state["agrid"].data[:-1, :-1, 1],
            state["agrid"].np,
        )
        set_c_grid_tile_border_area(
            xyz_dgrid[2:-2, 2:-2, :],
            xyz_agrid[2:-2, 2:-2, :],
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




class TranslateSetEta(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "npz": {
            "name": "npz",
            "dims": [],
            "units": "",
        },
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_sequential(self, inputs_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["ak"].data[:], state["bk"].data[:], state["ptop"].data[:] = set_eta(state["npz"])
        return state


class UtilVectors(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name":"ec1",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ec2": {
            "name":"ec2",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ew": {
            "name":"ew",
            "dims":[fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, 3, 2]
        },
        "es": {
            "name":"es",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, 3, 2]
        },
    }
    outputs: Dict[str, Any] = {
        "ec1": {
            "name":"ec1",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ec2": {
            "name":"ec2",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ew": {
            "name":"ew",
            "dims":[fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, 3, 2]
        },
        "es": {
            "name":"es",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, 3, 2]
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:,:,0], state["agrid"].data[:,:,1], state["grid"].np)
        state["ec1"].data[:], state["ec2"].data[:] = get_center_vector(xyz_dgrid, self.grid.halo, state["grid"].np)
        state["ew"].data[:] = calc_unit_vector_west(
            xyz_dgrid, xyz_agrid, self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["grid"].np
        )
        state["es"].data[:] = calc_unit_vector_south(
            xyz_dgrid, xyz_agrid, self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["grid"].np
        )
        return state


class TranslateTrigSg(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid3": {
            "name": "grid3",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, 3],
            "units": ""
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "ec1": {
            "name":"ec1",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ec2": {
            "name":"ec2",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:,:,0], state["agrid"].data[:,:,1], state["agrid"].np)
        state["cos_sg"].data[:], state["sin_sg"].data[:] = calculate_cos_sin_sg(
            state["grid3"].data[:], xyz_agrid, state["ec1"].data[:], state["ec2"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["agrid"].np
        )
        return state


class TranslateAAMCorrection(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "radians",
        },
        "l2c_u": {
            "name":"l2c_v",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM]
        },
    }
    outputs: Dict[str, Any] = {
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "radians",
        },
        "l2c_u": {
            "name":"l2c_v",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM]
        },
    }
    def compute_sequential(self, inputs_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)
    
    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        state["l2c_v"].data[:], state["l2c_u"].data[:] = calculate_l2c_vu(
            state["grid"].data[:], xyz_dgrid, self.grid.halo, state["grid"].np
        )
        return state


class TranslateMoreTrig(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "ee1": {
            "name": "ee1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": ""
        },
        "ee2": {
            "name": "ee2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
    }
    outputs: Dict[str, Any] = {
        "ee1": {
            "name": "ee1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": ""
        },
        "ee2": {
            "name": "ee2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:,:,0], state["agrid"].data[:,:,1], state["agrid"].np)
        state["cos_sg"].data[:], state["sin_sg"].data[:] = calculate_cos_sin_sg(
            state["grid3"].data[:], xyz_agrid, state["ec1"].data[:], state["ec2"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["agrid"].np
        )
        return state


class TranslateFixSgCorners(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        sg_corner_fix(
            state["cos_sg"].data[:], state["sin_sg"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["cos_sg"].np
        )
        return state


class TranslateDivgDel6(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
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
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }
    outputs: Dict[str, Any] = {
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
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
        state["divg_u"].data[:], state["divg_v"].data[:], state["del6_u"].data[:], state["del6_v"].data[:] = calculate_divg_del6(
            state["sin_sg"].data[:], state["sina_u"].data[:], state["sina_v"].data[:], 
            state["dx"].data[:], state["dy"].data[:], state["dxc"].data[:], state["dyc"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, state["sin_sg"].np
        )
        return state


class TranslateInitCubedtoLatLon(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name":"ec1",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ec2": {
            "name":"ec2",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
    }
    outputs: Dict[str, Any] = {
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": "",
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": "",
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    def compute_sequential(self, inputs_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)
    
    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["vlon"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, 3], ""
        )
        state["vlat"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, 3], ""
        )
        state["z11"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["z12"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["z21"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["z22"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["a11"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["a12"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["a21"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["a22"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )

        state["vlon"].data[:], state["vlat"].data[:] = unit_vector_lonlat(state["agrid"].data[:], state["agrid"].np)
        state["z11"].data[:], state["z12"].data[:], state["z21"].data[:], state["z22"].data[:] = calculate_grid_z(
            state["ec1"].data, state["ec2"].data, state["vlon"].data, state["vlat"].data[:], state["agrid"].np
        )
        state["a11"].data[:], state["a12"].data[:], state["a21"].data[:], state["a22"].data[:] = calculate_grid_a(
            state["z11"].data[:], state["z12"].data[:], state["z21"].data[:], state["z22"].data[:], state["sin_sg"].data[:], state["agrid"].np
        )
        return state


class TranslateEdgeFactors(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        state["edge_w"].data[:], state["edge_e"].data[:], state["edge_s"].data[:], state["edge_n"].data[:] = edge_factors(
            state["grid"].data[:], state["agrid"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, RADIUS, state["grid"].np
        )
        state["edge_vect_w"].data[:], state["edge_vect_e"].data[:], state["edge_vect_s"].data[:], state["edge_vect_n"].data[:] = efactor_a2c_v(
            state["grid"].data[:], state["agrid"].data[:], self.grid.halo, 
            communicator.tile.partitioner, communicator.tile.rank, RADIUS, state["grid"].np
        )
        return state


class TranslateInitGridUtils(ParallelTranslateGrid):

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
        "grid": {
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
    outputs: Dict[str, Any] = {
        "edge_s": {
            "name": "edge_south",
            "dims": [fv3util.X_DIM],
            "units": ""
        },
        "edge_n": {
            "name": "edge_north",
            "dims": [fv3util.X_DIM],
            "units": ""
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_DIM],
            "units": ""
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_DIM],
            "units": "ccc"
        },
        "edge_vect_s": {
            "name": "edge_vect_south",
            "dims": [fv3util.X_DIM],
            "units": ""
        },
        "edge_vect_n": {
            "name": "edge_vect_north",
            "dims": [fv3util.X_DIM],
            "units": ""
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": ""
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "ccc"
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg": {
            "name": "cos_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "sin_sg": {
            "name": "sin_sg",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 9],
            "units": ""
        },
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },"vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": "",
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, 3],
            "units": "",
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "da_min": {
            "name": "da_min",
            "dims": [],
            "units": "m^2"
        },
        "da_max": {
            "name": "da_max",
            "dims": [],
            "units": "m^2"
        },
        "da_min_c": {
            "name": "da_min_c",
            "dims": [],
            "units": "m^2"
        },
        "da_max_c": {
            "name": "da_max_c",
            "dims": [],
            "units": "m^2"
        },
        "ec1": {
            "name":"ec1",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ec2": {
            "name":"ec2",
            "dims":[fv3util.X_DIM, fv3util.Y_DIM, 3]
        },
        "ew": {
            "name":"ew",
            "dims":[fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, 3, 2]
        },
        "es": {
            "name":"es",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, 3, 2]
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local_eta(inputs))
        for i, state in enumerate(state_list):
            fill_corners_2d(
                state["grid"].data[:, :, :], self.grid, gridtype="B", direction="x"
            )
            state_list[i] = _compute_local_part2(state)
            
        

            
        return self.outputs_list_from_state_list(state_list)


    def _compute_local_eta(self, inputs):
        state = self.state_from_inputs(inputs)
        state["eta"] = set_eta(state)
        return state

    def _compute_local_part2(self, state):
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        center_vector1, center_vector2 = get_center_vector(xyz_dgrid, self.grid.halo)
        center_vector1[:self.grid.halo, :self.grid.halo, :] = 1.e8
        center_vector1[:self.grid.halo, -self.grid.halo:, :] = 1.e8
        center_vector1[-self.grid.halo:, :self.grid.halo, :] = 1.e8
        center_vector1[-self.grid.halo:, -self.grid.halo:, :] = 1.e8

        center_vector2[:self.grid.halo, :self.grid.halo, :] = 1.e8
        center_vector2[:self.grid.halo, -self.grid.halo:, :] = 1.e8
        center_vector2[-self.grid.halo:, :self.grid.halo, :] = 1.e8
        center_vector2[-self.grid.halo:, -self.grid.halo:, :] = 1.e8

        return state

    def _compute_outputs(self, state):
        outputs = self.outputs_from_state(state)
        return outputs
