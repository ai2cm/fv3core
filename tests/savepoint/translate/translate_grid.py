import functools
from math import cos
from typing import Any, Dict
from fv3core.utils.grid import GridIndexing
from fv3core import grid
import numpy
import copy
import fv3gfs.util as fv3util
from fv3core.utils.null_comm import NullComm
import fv3core.utils.global_config as global_config
import fv3core._config as spec
from fv3core.grid import (
    get_area,
    local_gnomonic_ed,
    great_circle_distance_along_axis,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    mirror_grid,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
    set_halo_nan,
    MetricTerms
)

from fv3core.utils import gt4py_utils as utils

from fv3core.grid.geometry import (
    get_center_vector,
    calc_unit_vector_west,
    calc_unit_vector_south,
    calculate_supergrid_cos_sin,
    calculate_l2c_vu,
    supergrid_corner_fix,
    calculate_divg_del6,
    unit_vector_lonlat,
    calculate_grid_z, 
    calculate_grid_a,
    edge_factors,
    efactor_a2c_v,
    calculate_trig_uv,
    generate_xy_unit_vectors
)
from fv3core.grid.eta import set_eta

from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM, CARTESIAN_DIM
from fv3core.utils import gt4py_utils as utils
from fv3core.testing.parallel_translate import ParallelTranslateGrid, _serialize_slice


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
        global_mirror_grid(
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
        for i,inputs in enumerate(inputs_list):
            state_list.append(self._compute_local(inputs, communicator_list[i].partitioner.tile, i))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, tile_partitioner, rank):
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
            tile_partitioner=tile_partitioner,
            rank = rank,
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
                self.rank_grids[i],
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
            communicator.rank,
            state["grid"].np,
        )
     
        set_tile_border_dxc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dx_cgrid"].data[3:-3, 3:-4],
            communicator.tile.partitioner,
            communicator.rank,
            state["grid"].np,
        )
        set_tile_border_dyc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dy_cgrid"].data[3:-4, 3:-3],
            communicator.tile.partitioner,
            communicator.rank,
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
        for state,grid in zip(state_list, self.rank_grids):
            fill_corners_2d(
                state["grid"].data[:, :, :], grid, gridtype="B", direction="x"
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
                self.rank_grids[i],
                gridtype="A",
                direction="x",
            )
            fill_corners_2d(
                state["agrid"].data[:, :, 1][:, :, None],
                self.rank_grids[i],
                gridtype="A",
                direction="y",
            )
            state_list[i] = self._compute_local_part2(state, self.rank_grids[i])
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

    def _compute_local_part2(self, state, grid):
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
            dx_agrid[:, :, None], dy_agrid[:, :, None], grid, vector=False
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
    inputs = {
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

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(npx=namelist.npx, npy=namelist.npy, npz=1, communicator=communicator,  backend=global_config.get_backend())
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)

    
    
    def compute_sequential(self, inputs_list, communicator_list):
        layout = spec.namelist.layout
        halo = self.grid.halo
        local_sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.grid.npx - 1,
            ny_tile=self.grid.npy - 1,
            nz=self.grid.npz,
            n_halo=halo,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=layout,
        )
        local_quantity_factory = fv3util.QuantityFactory.from_backend(
            local_sizer, backend=global_config.get_backend()

        )
       
        
        grid_dims =  [
            fv3util.X_INTERFACE_DIM,
            fv3util.Y_INTERFACE_DIM,
            LON_OR_LAT_DIM,
        ]
        shift_fac = 18
       
        state_list = []
        namelist = spec.namelist
       
        for i, inputs in enumerate(inputs_list):
            rank = communicator_list[i].rank
            partitioner =  communicator_list[i].partitioner
            tile_index = partitioner.tile_index(i)
            tile = partitioner.tile
           
            grid_section = local_quantity_factory.zeros(grid_dims, "radians", dtype=float,)
            grid_mirror_ew = local_quantity_factory.zeros(grid_dims, "radians", dtype=float,)
            grid_mirror_ns = local_quantity_factory.zeros(grid_dims, "radians", dtype=float,)
            grid_mirror_diag = local_quantity_factory.zeros(grid_dims, "radians", dtype=float,)
            
        
            local_west_edge = tile.on_tile_left(rank)
            local_east_edge = tile.on_tile_right(rank)
            local_south_edge = tile.on_tile_bottom(rank)
            local_north_edge =  tile.on_tile_top(rank)
            npx, npy, ndims  = tile.global_extent(grid_section)
            slice_x, slice_y = tile.subtile_slice(rank, grid_section.dims, (npx, npy), overlap=True)
            section_global_is = halo + slice_x.start
            section_global_js = halo + slice_y.start
            subtile_width_x = slice_x.stop - slice_x.start - 1
            subtile_width_y = slice_y.stop - slice_y.start - 1
            # compute for this rank
            local_gnomonic_ed( grid_section.view[:,:,0],  grid_section.view[:,:,1],  npx=npx,
                               west_edge=local_west_edge,
                               east_edge=local_east_edge,
                               south_edge=local_south_edge,
                               north_edge=local_north_edge,
                               global_is=section_global_is,
                               global_js=section_global_js,
                               np=grid_section.np, rank=rank)
            # Now compute for the mirrored ranks that'll be averaged
            j_subtile_index, i_subtile_index = partitioner.tile.subtile_index(i)
            ew_i_subtile_index = layout[0] - i_subtile_index - 1
            ns_j_subtile_index = layout[1] - j_subtile_index - 1
            ew_global_is = halo +  ew_i_subtile_index * subtile_width_x
            ns_global_js = halo +  ns_j_subtile_index * subtile_width_y

            # compute mirror in the east-west direction
            west_edge = True if local_east_edge else False
            east_edge = True if local_west_edge else False   
            local_gnomonic_ed(grid_mirror_ew.view[:,:,0],  grid_mirror_ew.view[:,:,1],  npx=npx,
                              west_edge=west_edge,
                              east_edge=east_edge,
                              south_edge=local_south_edge,
                              north_edge=local_north_edge,
                              global_is=ew_global_is,
                              global_js=section_global_js,
                              np=grid_section.np, rank=rank)

            # compute mirror in the north-south direction
            south_edge = True if local_north_edge else False
            north_edge = True if local_south_edge else False
            local_gnomonic_ed(grid_mirror_ns.view[:,:,0],  grid_mirror_ns.view[:,:,1],  npx=npx,
                              west_edge=local_west_edge,
                              east_edge=local_east_edge,
                              south_edge=south_edge,
                              north_edge=north_edge,
                              global_is=section_global_is,
                              global_js=ns_global_js,
                              np=grid_section.np, rank=rank)

            # compute mirror in the diagonal
            local_gnomonic_ed(grid_mirror_diag.view[:,:,0],  grid_mirror_diag.view[:,:,1],  npx=npx,
                              west_edge=west_edge,
                              east_edge=east_edge,
                              south_edge=south_edge,
                              north_edge=north_edge,
                              global_is=ew_global_is,
                              global_js=ns_global_js,
                              np=grid_section.np, rank=rank)
            
            # Mirror
            mirror_data = {'local': grid_section.data, 'east-west': grid_mirror_ew.data, 'north-south': grid_mirror_ns.data, 'diagonal': grid_mirror_diag.data}
            mirror_grid(mirror_data, tile_index, npx, npy, subtile_width_x+1,subtile_width_x+1, section_global_is, section_global_js, halo,grid_section.np,)
          
            # Shift the corner away from Japan
            # This will result in the corner close to east coast of China
            grid_section.view[:, :, 0] -= PI / shift_fac
            lon = grid_section.data[:, :, 0]
            lon[lon < 0] += 2 * PI
            grid_section.data[grid_section.np.abs(grid_section.data[:]) < 1e-10] = 0.0
            state_list.append({"grid": grid_section})
        req_list = []
      
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_halo_update(state["grid"], n_points=self.grid.halo)
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
       
        grid_indexers = []
        for i, state in enumerate(state_list):
            grid_indexers.append(GridIndexing.from_sizer_and_communicator(local_sizer, communicator_list[i]))
            fill_corners_2d(
                state["grid"].data[:, :, :], grid_indexers[i], gridtype="B", direction="x"
            )
            state_list[i] = state

       
        #calculate d-grid cell side lengths
        for i, state in enumerate(state_list):
            self._compute_local_dxdy(state, local_quantity_factory)
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
        for grid_indexer, state, grid in zip(grid_indexers, state_list, self.rank_grids):
            state["dx"].data[state["dx"].data < 0] *= -1
            state["dy"].data[state["dy"].data < 0] *= -1
            fill_corners_dgrid(
                state["dx"].data[:, :, None],
                state["dy"].data[:, :, None],
                grid_indexer,
                vector=False,
            )
        

        #Set up lat-lon a-grid, calculate side lengths on a-grid
        for i, state in enumerate(state_list):
            self._compute_local_agrid_part1(state, local_quantity_factory)
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
                grid_indexers[i],
                gridtype="A",
                direction="x",
            )
            fill_corners_2d(
                state["agrid"].data[:, :, 1][:, :, None],
                grid_indexers[i],
                gridtype="A",
                direction="y",
            )
            self._compute_local_agrid_part2(state, local_quantity_factory,grid_indexers[i] )
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
            self._compute_local_areas_pt1(state, communicator_list[i], local_quantity_factory)
            

        #Finish c-grid areas, calculate sidelengths on the c-grid
        for i, state in enumerate(state_list):
            self._compute_local_areas_pt2(state, communicator_list[i])
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_vector_halo_update(
                    state["dx_cgrid"], state["dy_cgrid"], n_points=self.grid.halo
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            req.wait()
        for grid_indexer,state, grid in zip(grid_indexers,state_list, self.rank_grids):
            #TODO: Add support for unsigned vector halo updates instead of handling ad-hoc here
            state["dx_cgrid"].data[state["dx_cgrid"].data < 0] *= -1
            state["dy_cgrid"].data[state["dy_cgrid"].data < 0] *= -1

            #TODO: fix issue with interface dimensions causing validation errors
            fill_corners_cgrid(
                state["dx_cgrid"].data[:, :, None],
                state["dy_cgrid"].data[:, :, None],
                grid_indexer,
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
                grid_indexers[i],
                gridtype="B",
                direction="x",
            )
        
        return self.outputs_list_from_state_list(state_list)


    def _compute_local_dxdy(self, state, local_quantity_factory):
        state["dx"] = local_quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        state["dy"] = local_quantity_factory.zeros(
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
      


    def _compute_local_agrid_part1(self, state, local_quantity_factory):
        state["agrid"] = local_quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM], "radians"
        )
        lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        agrid_lon, agrid_lat = lon_lat_corner_to_cell_center(lon, lat, state["grid"].np)
        state["agrid"].data[:-1, :-1, 0], state["agrid"].data[:-1, :-1, 1] = (
            agrid_lon,
            agrid_lat,
        )
       

    def _compute_local_agrid_part2(self, state, local_quantity_factory, grid_indexer):
        state["dx_agrid"] = local_quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_agrid"] = local_quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dx_cgrid"] = local_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_cgrid"] = local_quantity_factory.zeros(
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
            dx_agrid[:, :, None], dy_agrid[:, :, None], grid_indexer, vector=False
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
        #outputs = self.allocate_output_state()
        #for name in ("dx_agrid", "dy_agrid"):
        #    state[name] = outputs[name]
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

      


    def _compute_local_areas_pt1(self, state, communicator, local_quantity_factory):
        state["area"] = local_quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m^2"
        )
        state["area"].data[:, :] = -1.e8
        state["area_cgrid"] = local_quantity_factory.zeros(
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
            tile_partitioner=communicator.tile.partitioner,
            rank=communicator.rank,
            radius=RADIUS,
            np=state["grid"].np,
        )
       
# rank = 0  diff 0.0360107421875,  diff 0.0721435546875
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
            communicator.rank,
            state["grid"].np,
        )
        set_tile_border_dxc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dx_cgrid"].data[3:-3, 3:-4],
            communicator.tile.partitioner,
            communicator.rank,
            state["grid"].np,
        )
        set_tile_border_dyc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dy_cgrid"].data[3:-4, 3:-3],
            communicator.tile.partitioner,
            communicator.rank,
            state["grid"].np,
        )
       


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

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["ks"], state["ptop"], state["ak"].data[:], state["bk"].data[:] = set_eta(state["npz"])
        return state


class TranslateUtilVectors(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2" : {
                "kend": 2,
                "kaxis": 0,
            },
            "es1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "es2" : {
                "kend": 2,
                "kaxis": 0,
            },
        }
       

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
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name":"ec2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name":"ew1",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name":"ew2",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name":"es1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name":"es2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ec1": {
            "name":"ec1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name":"ec2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name":"ew1",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name":"ew2",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name":"es1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name":"es2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        outputs_list = []
        # for state in state_list:
        #     outputs_list.append(self.outputs_from_state(state))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        # TODO why is the not necessary?
        #fill_corners_2d(
        #   state["grid"].data[:, :, :], self.grid, gridtype="B", direction="x"
        #)
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:-1,:-1,0], state["agrid"].data[:-1,:-1,1], state["grid"].np)
        
        state["ec1"].data[:-1,:-1,:3], state["ec2"].data[:-1,:-1,:3] = get_center_vector(xyz_dgrid, self.grid.grid_type, self.grid.halo,
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )
        state["ew1"].data[1:-1,:-1,:3], state["ew2"].data[1:-1,:-1,:3] = calc_unit_vector_west(
            xyz_dgrid, xyz_agrid, self.grid.grid_type, self.grid.halo, 
            communicator.partitioner.tile, communicator.rank, state["grid"].np
        )
        state["es1"].data[:-1,1:-1,:3], state["es2"].data[:-1,1:-1,:3] = calc_unit_vector_south(
            xyz_dgrid, xyz_agrid, self.grid.grid_type, self.grid.halo, 
            communicator.partitioner.tile, communicator.rank, state["grid"].np
        )
        return state


class TranslateTrigSg(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2" : {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": ""
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "ec1": {
            "name":"ec1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name":"ec2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
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
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["agrid"].np)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:-1,:-1,0], state["agrid"].data[:-1,:-1,1], state["agrid"].np)
        # csgs = state["cos_sg1"].data[:].shape
        # print(csgs, state["grid"].data[:].shape, state["agrid"].data[:].shape)
        cos_sg, sin_sg = calculate_supergrid_cos_sin(
            xyz_dgrid, xyz_agrid, state["ec1"].data[:-1, :-1], state["ec2"].data[:-1, :-1], self.grid.grid_type, self.grid.halo, 
            communicator.tile.partitioner, communicator.rank, state["agrid"].np
        )
        for i in range(1,10):
            state[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:,:,i-1]
            state[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:,:,i-1]
        return state


class TranslateAAMCorrection(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,

        },
        "l2c_u": {
            "name":"l2c_u",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }
    outputs: Dict[str, Any] = {
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name":"l2c_u",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)
    
    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        nhalo = self.grid.halo
        state["l2c_v"].data[nhalo:-nhalo, nhalo:-nhalo-1], state["l2c_u"].data[nhalo:-nhalo-1, nhalo:-nhalo] = calculate_l2c_vu(
            state["grid"].data[:], nhalo, state["grid"].np
        )
        return state


class TranslateMoreTrig(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ee1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2" : {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
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
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        cos_sg = []
        sin_sg = []
        for i in range(1, 10):
            cos_sg.append(state[f"cos_sg{i}"].data[:-1, :-1])
            sin_sg.append(state[f"sin_sg{i}"].data[:-1, :-1])
        cos_sg = state["grid"].np.array(cos_sg).transpose([1,2,0])
        sin_sg = state["grid"].np.array(sin_sg).transpose([1,2,0])
        nhalo = self.grid.halo
        state["ee1"].data[nhalo:-nhalo, nhalo:-nhalo, :], state["ee2"].data[nhalo:-nhalo, nhalo:-nhalo, :] = generate_xy_unit_vectors(xyz_dgrid, nhalo, communicator.tile.partitioner, communicator.rank, state["grid"].np)
        state["cosa"].data[:, :], state["sina"].data[:, :], state["cosa_u"].data[:, :-1], state["cosa_v"].data[:-1, :], state["cosa_s"].data[:-1, :-1], state["sina_u"].data[:, :-1], state["sina_v"].data[:-1, :], state["rsin_u"].data[:, :-1], state["rsin_v"].data[:-1, :], state["rsina"].data[nhalo:-nhalo, nhalo:-nhalo], state["rsin2"].data[:-1, :-1] = calculate_trig_uv(xyz_dgrid, cos_sg, sin_sg, nhalo, 
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )
        return state


class TranslateFixSgCorners(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
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
        cos_sg = []
        sin_sg = []
        for i in range(1, 10):
            cos_sg.append(state[f"cos_sg{i}"].data[:-1, :-1])
            sin_sg.append(state[f"sin_sg{i}"].data[:-1, :-1])
        cos_sg = state["cos_sg1"].np.array(cos_sg).transpose(1, 2, 0)
        sin_sg = state["cos_sg1"].np.array(sin_sg).transpose(1, 2, 0)
        supergrid_corner_fix(
            cos_sg, sin_sg, self.grid.halo, 
            communicator.tile.partitioner, communicator.rank
        )
        for i in range(1,10):
            state[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:,:,i-1]
            state[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:,:,i-1]
        return state


class TranslateDivgDel6(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
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
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
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
        sin_sg = []
        for i in range(1, 5):
            sin_sg.append(state[f"sin_sg{i}"].data[:-1, :-1])
        sin_sg = state["sin_sg1"].np.array(sin_sg).transpose(1, 2, 0)
        state["divg_u"].data[:-1, :], state["divg_v"].data[:, :-1], state["del6_u"].data[:-1, :], state["del6_v"].data[:, :-1] = calculate_divg_del6(
            sin_sg, state["sina_u"].data[:,:-1], state["sina_v"].data[:-1,:], 
            state["dx"].data[:-1, :], state["dy"].data[:, :-1], state["dx_cgrid"].data[:, :-1], state["dy_cgrid"].data[:-1, :], self.grid.halo, 
            communicator.tile.partitioner, communicator.rank
        )
        return state


class TranslateInitCubedtoLatLon(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2" : {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name":"ec1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name":"ec2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
    }
    outputs: Dict[str, Any] = {
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
    }
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)
    
    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["vlon"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["vlat"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
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

        state["vlon"].data[:-1, :-1], state["vlat"].data[:-1, :-1] = unit_vector_lonlat(state["agrid"].data[:-1, :-1], state["agrid"].np)
        state["z11"].data[:-1, :-1], state["z12"].data[:-1, :-1], state["z21"].data[:-1, :-1], state["z22"].data[:-1, :-1] = calculate_grid_z(
            state["ec1"].data[:-1, :-1], state["ec2"].data[:-1, :-1], state["vlon"].data[:-1, :-1], state["vlat"].data[:-1, :-1], state["agrid"].np
        )
        state["a11"].data[:-1, :-1], state["a12"].data[:-1, :-1], state["a21"].data[:-1, :-1], state["a22"].data[:-1, :-1] = calculate_grid_a(
            state["z11"].data[:-1, :-1], state["z12"].data[:-1, :-1], state["z21"].data[:-1, :-1], state["z22"].data[:-1, :-1], state["sin_sg5"].data[:-1, :-1]
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
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
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
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
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
        nhalo = self.grid.halo
        state["edge_w"].data[nhalo:-nhalo], state["edge_e"].data[nhalo:-nhalo], state["edge_s"].data[nhalo:-nhalo], state["edge_n"].data[nhalo:-nhalo] = edge_factors(
            state["grid"].data[:], state["agrid"].data[:-1, :-1], self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, RADIUS, state["grid"].np
        )
        state["edge_vect_w"].data[:-1], state["edge_vect_e"].data[:-1], state["edge_vect_s"].data[:-1], state["edge_vect_n"].data[:-1] = efactor_a2c_v(
            state["grid"], state["agrid"].data[:-1, :-1], self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, RADIUS, state["grid"].np
        )
        return state


class TranslateInitGridUtils(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2" : {
                "kend": 2,
                "kaxis": 0,
            },
            "es1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "es2" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ee1" : {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2" : {
                "kend": 2,
                "kaxis": 0,
            },
        }

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
        "npz": {
            "name": "npz",
            "dims": [],
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
        "ec1": {
            "name":"ec1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name":"ec2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name":"ew1",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name":"ew2",
            "dims":[CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name":"es1",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name":"es2",
            "dims":[CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name":"l2c_u",
            "dims":[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": ""
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": ""
        },
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
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
        "da_min": {
            "name": "da_min",
            "dims": [],
            "units": "m^2",
        },
        "da_min_c": {
            "name": "da_min_c",
            "dims": [],
            "units": "m^2",
        },
        "da_max": {
            "name": "da_min",
            "dims": [],
            "units": "m^2",
        },
        "da_max_c": {
            "name": "da_min_c",
            "dims": [],
            "units": "m^2",
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
            state_list[i] = self._compute_local_part_1(state, communicator_list[i])
        
        #TODO: omega and norm_vect computation if needed
        
        for i, state in enumerate(state_list):
            state_list[i] = self._compute_local_part2(state, communicator_list[i])
        
        #TODO: implement allreduce to get da_min, da_max, da_min_c, and da_max_c
        #temporary hack is possible by looping over ranks here:
        min_da = []
        max_da = []
        min_da_c = []
        max_da_c = []
        for state in state_list:
            min_da.append(state["grid"].np.min(state["area"].data[:]))
            max_da.append(state["grid"].np.max(state["area"].data[:]))
            min_da_c.append(state["grid"].np.min(state["area_cgrid"].data[:]))
            max_da_c.append(state["grid"].np.max(state["area_cgrid"].data[:]))
        da_min = min(min_da)
        da_max = max(max_da)
        da_min_c = min(min_da_c)
        da_max_c = max(max_da_c)
        for i, state in enumerate(state_list):
            state["da_min"] = self.grid.quantity_factory.zeros(
                [], ""
            )
            state["da_min"] = da_min
            state["da_max"] = self.grid.quantity_factory.zeros(
                [], ""
            )
            state["da_max"] = da_max
            state["da_min_c"] = self.grid.quantity_factory.zeros(
                [], ""
            )
            state["da_min_c"] = da_min_c
            state["da_max_c"] = self.grid.quantity_factory.zeros(
                [], ""
            )
            state["da_max_c"] = da_max_c

        for i, state in enumerate(state_list):
            state_list[i] = self._compute_local_edges(state, communicator_list[i])

        return self.outputs_list_from_state_list(state_list)


    def _compute_local_eta(self, inputs):
        state = self.state_from_inputs(inputs)
        npz = state["npz"]
        state["ks"] = self.grid.quantity_factory.zeros(
            [], ""
        )
        state["ptop"] = self.grid.quantity_factory.zeros(
            [], "mb"
        )
        state["ak"] = self.grid.quantity_factory.zeros(
            [fv3util.Z_INTERFACE_DIM], "mb"
        )
        state["bk"] = self.grid.quantity_factory.zeros(
            [fv3util.Z_INTERFACE_DIM], "mb"
        )
        state["ks"], state["ptop"], state["ak"].data[:], state["bk"].data[:] = set_eta(npz)
        return state

    def _compute_local_part_1(self, state, communicator):
        nhalo = self.grid.halo
        state["ec1"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["ec2"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["ew1"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["ew2"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["es1"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        state["es2"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        xyz_dgrid = lon_lat_to_xyz(state["grid"].data[:,:,0], state["grid"].data[:,:,1], state["grid"].np)
        xyz_agrid = lon_lat_to_xyz(state["agrid"].data[:-1,:-1,0], state["agrid"].data[:-1,:-1,1], state["grid"].np)
        
        state["ec1"].data[:-1,:-1,:3], state["ec2"].data[:-1,:-1,:3] = get_center_vector(xyz_dgrid, self.grid.grid_type, nhalo,
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )
        state["ew1"].data[1:-1,:-1,:3], state["ew2"].data[1:-1,:-1,:3] = calc_unit_vector_west(
            xyz_dgrid, xyz_agrid, self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )
        state["es1"].data[:-1,1:-1,:3], state["es2"].data[:-1,1:-1,:3] = calc_unit_vector_south(
            xyz_dgrid, xyz_agrid, self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )

        cos_sg, sin_sg = calculate_supergrid_cos_sin(
            xyz_dgrid, xyz_agrid, state["ec1"].data[:-1, :-1], state["ec2"].data[:-1, :-1], self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, state["agrid"].np
        )
        for i in range(1,10):
            state[f"cos_sg{i}"] = self.grid.quantity_factory.zeros(
                [fv3util.X_DIM, fv3util.Y_DIM], ""
            )
            state[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:,:,i-1]
            state[f"sin_sg{i}"] = self.grid.quantity_factory.zeros(
                [fv3util.X_DIM, fv3util.Y_DIM], ""
            )
            state[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:,:,i-1]

        state["l2c_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        state["l2c_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["l2c_v"].data[nhalo:-nhalo, nhalo:-nhalo-1], state["l2c_u"].data[nhalo:-nhalo-1, nhalo:-nhalo] = calculate_l2c_vu(
            state["grid"].data[:], nhalo, state["grid"].np
        )

        state["cosa_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        state["cosa_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["cosa_s"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["sina_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        state["sina_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["rsin_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        state["rsin_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["rsina"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["rsin2"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], ""
        )
        state["cosa"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["sina"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["ee1"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        state["ee2"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        state["ee1"].data[nhalo:-nhalo, nhalo:-nhalo, :], state["ee2"].data[nhalo:-nhalo, nhalo:-nhalo, :] = generate_xy_unit_vectors(xyz_dgrid, nhalo, communicator.tile.partitioner, communicator.rank, state["grid"].np)
        state["cosa"].data[:, :], state["sina"].data[:, :], state["cosa_u"].data[:, :-1], state["cosa_v"].data[:-1, :], state["cosa_s"].data[:-1, :-1], state["sina_u"].data[:, :-1], state["sina_v"].data[:-1, :], state["rsin_u"].data[:, :-1], state["rsin_v"].data[:-1, :], state["rsina"].data[nhalo:-nhalo, nhalo:-nhalo], state["rsin2"].data[:-1, :-1] = calculate_trig_uv(xyz_dgrid, cos_sg, sin_sg, nhalo, 
            communicator.tile.partitioner, communicator.rank, state["grid"].np
        )

        supergrid_corner_fix(
            cos_sg, sin_sg, nhalo, 
            communicator.tile.partitioner, communicator.rank
        )
        for i in range(1,10):
            state[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:,:,i-1]
            state[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:,:,i-1]

        return state

    def _compute_local_part2(self, state, communicator):
        nhalo = self.grid.halo
        state["del6_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["del6_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        state["divg_u"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        state["divg_v"] = self.grid.quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        sin_sg = []
        for i in range(1, 5):
            sin_sg.append(state[f"sin_sg{i}"].data[:-1, :-1])
        sin_sg = state["sin_sg1"].np.array(sin_sg).transpose(1, 2, 0)
        state["divg_u"].data[:-1, :], state["divg_v"].data[:, :-1], state["del6_u"].data[:-1, :], state["del6_v"].data[:, :-1] = calculate_divg_del6(
            sin_sg, state["sina_u"].data[:,:-1], state["sina_v"].data[:-1,:], 
            state["dx"].data[:-1, :], state["dy"].data[:, :-1], state["dx_cgrid"].data[:, :-1], state["dy_cgrid"].data[:-1, :], nhalo, 
            communicator.tile.partitioner, communicator.rank
        )
        state["vlon"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        state["vlat"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
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

        state["vlon"].data[:-1, :-1], state["vlat"].data[:-1, :-1] = unit_vector_lonlat(state["agrid"].data[:-1, :-1], state["agrid"].np)
        state["z11"].data[:-1, :-1], state["z12"].data[:-1, :-1], state["z21"].data[:-1, :-1], state["z22"].data[:-1, :-1] = calculate_grid_z(
            state["ec1"].data[:-1, :-1], state["ec2"].data[:-1, :-1], state["vlon"].data[:-1, :-1], state["vlat"].data[:-1, :-1], state["agrid"].np
        )
        state["a11"].data[:-1, :-1], state["a12"].data[:-1, :-1], state["a21"].data[:-1, :-1], state["a22"].data[:-1, :-1] = calculate_grid_a(
            state["z11"].data[:-1, :-1], state["z12"].data[:-1, :-1], state["z21"].data[:-1, :-1], state["z22"].data[:-1, :-1], state["sin_sg5"].data[:-1, :-1]
        )

        return state

    def _compute_local_edges(self, state, communicator):
        nhalo = self.grid.halo
        state["edge_s"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM], ""
        )
        state["edge_n"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM], ""
        )
        state["edge_e"] = self.grid.quantity_factory.zeros(
            [fv3util.Y_DIM], ""
        )
        state["edge_w"] = self.grid.quantity_factory.zeros(
            [fv3util.Y_DIM], ""
        )
        state["edge_w"].data[nhalo:-nhalo], state["edge_e"].data[nhalo:-nhalo], state["edge_s"].data[nhalo:-nhalo], state["edge_n"].data[nhalo:-nhalo] = edge_factors(
            state["grid"].data[:], state["agrid"].data[:-1, :-1], self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, RADIUS, state["grid"].np
        )

        state["edge_vect_s"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM], ""
        )
        state["edge_vect_n"] = self.grid.quantity_factory.zeros(
            [fv3util.X_DIM], ""
        )
        state["edge_vect_e"] = self.grid.quantity_factory.zeros(
            [fv3util.Y_DIM], ""
        )
        state["edge_vect_w"] = self.grid.quantity_factory.zeros(
            [fv3util.Y_DIM], ""
        )
        state["edge_vect_w"].data[:-1], state["edge_vect_e"].data[:-1], state["edge_vect_s"].data[:-1], state["edge_vect_n"].data[:-1] = efactor_a2c_v(
            state["grid"], state["agrid"].data[:-1, :-1], self.grid.grid_type, nhalo, 
            communicator.tile.partitioner, communicator.rank, RADIUS, state["grid"].np
        )
        return state
