import functools
from fv3core.grid.utils import get_center_vector # noqa: F401
from typing import Any, Dict
from fv3core.utils.grid import GridIndexing
import fv3gfs.util as fv3util
import fv3core.utils.global_config as global_config
import fv3core._config as spec
from fv3core.grid import (
    get_area,
    gnomonic_grid, local_gnomonic_ed,
    great_circle_distance_along_axis,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    mirror_grid,local_mirror_grid,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
    set_halo_nan,
    MetricTerms
)
from fv3core.utils import gt4py_utils as utils
from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM
from fv3core.testing.parallel_translate import ParallelTranslateGrid


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
    }
    """
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
    """
    
    def __init__(self, grids):
        super().__init__(grids)
        self.ignore_near_zero_errors = {}
        self.ignore_near_zero_errors["grid"] = True

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms(self.grid,grid_type=self.grid.grid_type, layout=self.layout, npx=namelist.npx, npy=namelist.npy, npz=namelist.npz, communicator=communicator,  backend=global_config.get_backend())
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)

    
    
    def compute_sequential(self, inputs_list, communicator_list):
        layout = spec.namelist.layout
        local_sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.grid.npx - 1,
            ny_tile=self.grid.npy - 1,
            nz=self.grid.npz,
            n_halo=3,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=layout,
        )
        local_quantity_factory = fv3util.QuantityFactory.from_backend(
            local_sizer, backend=global_config.get_backend()
        )
        global_sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.grid.npx - 1,
            ny_tile=self.grid.npy - 1,
            nz=self.grid.npz,
            n_halo=3,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=(1,1),
        )
        global_quantity_factory = fv3util.QuantityFactory.from_backend(
            global_sizer, backend=global_config.get_backend()
        )
        #Set up initial lat-lon d-grid
        shift_fac = 18
        grid_global = global_quantity_factory.zeros(
            [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "radians",
            dtype=float,
        )
        lon_global = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        lat_global = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        state_list = []
        sections = {}
        compare = True
        for i, inputs in enumerate(inputs_list):
            partitioner =  communicator_list[i].partitioner
            old_grid = self.rank_grids[i]
            tile_index = partitioner.tile_index(i)
            grid_dims =  [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
            ]
            grid_section = local_quantity_factory.zeros(
                grid_dims,
                "radians",
                dtype=float,
            )
            grid_mirror_ew = local_quantity_factory.zeros(
                grid_dims,
                "radians",
                dtype=float,
            )
            grid_mirror_ns = local_quantity_factory.zeros(
                grid_dims,
                "radians",
                dtype=float,
            )
            grid_mirror_diag = local_quantity_factory.zeros(
                grid_dims,
                "radians",
                dtype=float,
            )
            #print('global extent', partitioner.global_extent(grid_section.metadata))
            #print('local extent', partitioner.tile.subtile_extent(grid_section.metadata))
            #print('subtile index', i, partitioner.tile.subtile_index(i), partitioner.tile.on_tile_left(i), partitioner.tile.on_tile_right(i), partitioner.tile.on_tile_bottom(i), partitioner.tile.on_tile_top(i))
        
            #local_gnomonic_ed(lon.view[:], lat.view[:], old_grid, lon.np)
            #print("\nmain", old_grid.rank, old_grid.west_edge,old_grid.east_edge,old_grid.sw_corner,old_grid.se_corner,old_grid.nw_corner,old_grid.ne_corner, old_grid.global_is, old_grid.global_js)
            local_gnomonic_ed( grid_section.view[:,:,0],  grid_section.view[:,:,1],  npx=old_grid.npx,
                               west_edge=old_grid.west_edge,
                               east_edge=old_grid.east_edge,
                               south_edge=old_grid.south_edge,
                               north_edge=old_grid.north_edge,
                               global_is=old_grid.global_is,
                               global_js=old_grid.global_js,
                               np=grid_section.np, rank=old_grid.rank)
            j_subtile_index, i_subtile_index = partitioner.tile.subtile_index(i)
            ew_i_subtile_index = layout[0] - i_subtile_index - 1
            ns_j_subtile_index = layout[1] - j_subtile_index - 1
            west_edge = True if old_grid.east_edge else False
            east_edge = True if old_grid.west_edge else False
            
            global_is = old_grid.local_to_global_1d(old_grid.is_, ew_i_subtile_index, old_grid.subtile_width_x)
            #print('ew', old_grid.rank, west_edge,east_edge,sw_corner,se_corner,nw_corner,ne_corner, global_is, old_grid.global_js, ew_i_subtile_index,  ns_j_subtile_index)
    
            local_gnomonic_ed(grid_mirror_ew.view[:,:,0],  grid_mirror_ew.view[:,:,1],  npx=old_grid.npx,
                              west_edge=west_edge,
                              east_edge=east_edge,
                              south_edge=old_grid.south_edge,
                              north_edge=old_grid.north_edge,
                              global_is=global_is,
                              global_js=old_grid.global_js,
                              np=grid_section.np, rank=old_grid.rank)
          
            south_edge = True if old_grid.north_edge else False
            north_edge = True if old_grid.south_edge else False
            global_js = old_grid.local_to_global_1d(old_grid.js, ns_j_subtile_index, old_grid.subtile_width_x)
            #print('nw', old_grid.rank, old_grid.west_edge,old_grid.east_edge,sw_corner,se_corner,nw_corner,ne_corner, old_grid.global_is, global_js, ew_i_subtile_index, ns_j_subtile_index,  ew_i_subtile_index,  ns_j_subtile_index)
            local_gnomonic_ed(grid_mirror_ns.view[:,:,0],  grid_mirror_ns.view[:,:,1],  npx=old_grid.npx,
                              west_edge=old_grid.west_edge,
                              east_edge=old_grid.east_edge,
                              south_edge=south_edge,
                              north_edge=north_edge,
                              global_is=old_grid.global_is,
                              global_js=global_js,
                              np=grid_section.np, rank=old_grid.rank)

            #print('diag', old_grid.rank, west_edge,east_edge,sw_corner,se_corner,nw_corner,ne_corner, global_is, global_js, ew_i_subtile_index, ns_j_subtile_index)
    
            local_gnomonic_ed(grid_mirror_diag.view[:,:,0],  grid_mirror_diag.view[:,:,1],  npx=old_grid.npx,
                              west_edge=west_edge,
                              east_edge=east_edge,
                              south_edge=south_edge,
                              north_edge=north_edge,
                              global_is=global_is,
                              global_js=global_js,
                              np=grid_section.np, rank=old_grid.rank)
            #local_mirror_grid(grid_global.data,old_grid,tile_index, grid_global.np,)
            
            if not compare:
                grid_global.data[old_grid.global_is:old_grid.global_ie+2, old_grid.global_js:old_grid.global_je+2, :, tile_index] = grid_section.data[old_grid.is_:old_grid.ie+2, old_grid.js:old_grid.je+2, :]
            sections[old_grid.rank] = grid_section
        if compare:
            gnomonic_grid(self.grid.grid_type,lon_global.view[:],lat_global.view[:],lon_global.np,)
            grid_global.view[:, :, 0, 0] = lon_global.view[:]
            grid_global.view[:, :, 1, 0] = lat_global.view[:]
            """
            for rank in range(min(9, len(inputs_list))):
                old_grid =  self.rank_grids[rank]
                section = sections[rank]
                for i in range(old_grid.nic+1):
                    for j in range(old_grid.njc+1):
                        g = grid_global.data[old_grid.global_is + i, old_grid.global_js+j, 0, 0]
                        glat = grid_global.data[old_grid.global_is + i, old_grid.global_js+j, 1, 0]
                        s = section.data[old_grid.is_ + i, old_grid.js + j, 0]
                        slat = section.data[old_grid.is_ + i, old_grid.js + j, 1]
                        if not (abs(g - s) < 1e-14 and  abs(glat - slat) < 1e-14):
                            print(rank, i, j, g, s, g == s, glat, slat, glat == slat)
            """
            mirror_grid(grid_global.data,self.grid.halo,self.grid.npx,self.grid.npy,grid_global.np,)
            for rank in range(min(9, len(inputs_list))):
                old_grid =  self.rank_grids[rank]
                section = sections[rank]
                for i in range(old_grid.nic+1):
                    for j in range(old_grid.njc+1):
                        g = grid_global.data[old_grid.global_is + i, old_grid.global_js+j, 0, 0]
                        glat = grid_global.data[old_grid.global_is + i, old_grid.global_js+j, 1, 0]
                        s = section.data[old_grid.is_ + i, old_grid.js + j, 0]
                        slat = section.data[old_grid.is_ + i, old_grid.js + j, 1]
                        #if not (abs(g - s) < 1e-16 and  abs(glat - slat) < 1e-16):
                        if not (g == s and glat == slat):
                            print(rank, i, j, g, s, g == s, glat, slat, glat == slat)
        
        #
        #mirror_grid(grid_global.data,self.grid.halo,self.grid.npx,self.grid.npy,grid_global.np,)
        # Shift the corner away from Japan
        # This will result in the corner close to east coast of China
        grid_global.view[:, :, 0, :] -= PI / shift_fac
        lon = grid_global.data[:, :, 0, :]
        lon[lon < 0] += 2 * PI
        grid_global.data[grid_global.np.abs(grid_global.data[:]) < 1e-10] = 0.0
        #state_list.append({"grid": grid_global})
        # more global copying
        #npx = self.grid.npx
        #npy = self.grid.npy
     
        #state_list = []
        for i, inputs in enumerate(inputs_list):
            rank_grid = self.rank_grids[i]
            tile_index = communicator_list[i].partitioner.tile_index(i)
            this_grid = local_quantity_factory.zeros(
                dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
                units="radians",dtype=float
            )
            this_grid.data[rank_grid.is_:rank_grid.ie+2, rank_grid.js:rank_grid.je+2, :] = grid_global.data[rank_grid.global_is:rank_grid.global_ie+2, rank_grid.global_js:rank_grid.global_je+2, :, tile_index]
        
            
            state_list.append({"grid": this_grid})
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
