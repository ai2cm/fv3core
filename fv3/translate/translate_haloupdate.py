from .parallel_translate import ParallelTranslate,_serialize_slice
from .translate import TranslateFortranData2Py
import fv3util
from fv3.utils import gt4py_utils as utils
import logging
from mpi4py import MPI

logger = logging.getLogger("fv3ser")


class TranslateHaloUpdate(ParallelTranslate):

    inputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": utils.halo,
        }
    }
    halo_update_varname = "air_temperature"
    def __init__(self, grid):
        super().__init__(grid)
        #self.compute_func = yppm.compute_flux
        #self.in_vars["data_vars"] = {
        #    "array": {}
        #}
        #self.in_vars["parameters"] = []
        #self.out_vars = {
        #    "array": {
        #    }
        #}

    def compute_parallel(self, inputs, rank_communicator):
        name = "array"
        self._base.make_storage_data_input_vars(inputs)
        properties = self.inputs[name]
        origin, extent = self.get_origin_and_extent(
            properties["dims"],
            inputs[name].shape,
            self.grid.npx,
            self.grid.npy,
            self.grid.npz,
            utils.halo,
            self.layout,
        )
        print('rank',rank_communicator.rank,'origin', origin, 'extent', extent, 'shape',inputs[name].shape)
       
        array = fv3util.Quantity(
            inputs[name],
            dims=properties["dims"],
            units= properties["units"],
            origin=origin,
            extent=extent,
        )
        print('rank',rank_communicator.rank,'shape',array.data.shape, 'data', array.data[0, 4, 0])
        for i in range(array.data.shape[0]):
            for j in range(array.data.shape[1]):
                for k in range(array.data.shape[2]):
                    if array.data[i, j, k] == 70.6421745627577:
                        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&FOUND', rank_communicator.rank, i, j, k)
        rank_communicator.start_halo_update(array, n_points=utils.halo)
       
        rank_communicator.finish_halo_update(array, n_points=utils.halo)
        print('after, rank',rank_communicator.rank, array.data[0, 4, 0])
        for i in range(array.data.shape[0]):
            for j in range(array.data.shape[1]):
                for k in range(array.data.shape[2]):
                    if array.data[i, j, k] == 70.6421745627577:
                        print('AFTER&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&FOUND', rank_communicator.rank, i, j, k)
        return {"array": array.data[ _serialize_slice(array, utils.halo)]}

    def get_origin_and_extent(self, dims, shape, npx, npy, npz, n_halo, layout):
        nx_rank = (npx - 1) / layout[1]
        ny_rank = (npy - 1) / layout[0]
        dim_lengths = {
            fv3util.X_DIM: nx_rank,
            fv3util.X_INTERFACE_DIM: nx_rank + 1,
            fv3util.Y_DIM: ny_rank,
            fv3util.Y_INTERFACE_DIM: ny_rank + 1,
            fv3util.Z_DIM: npz,
            fv3util.Z_INTERFACE_DIM: npz + 1,
        }
        origin = []
        extent = []
        for dim, current_length in zip(dims, shape):
            extent.append(int(dim_lengths.get(dim, current_length)))
            if dim in fv3util.HORIZONTAL_DIMS:
                halo = n_halo
            else:
                halo = 0
            origin.append(int(halo))
        return origin, extent

    '''
    def __init__(self, rank_grid):
        super(TranslateHaloUpdate, self).__init__(rank_grid)
    
    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"starting on {communicator.rank}")
            communicator.start_halo_update(
                state[self.halo_update_varname], n_points=utils.halo
            )
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"finishing on {communicator.rank}")
            communicator.finish_halo_update(
                state[self.halo_update_varname], n_points=utils.halo
            )
        return self.outputs_list_from_state_list(state_list)
    '''
'''
class TranslateHaloUpdate_2(TranslateHaloUpdate):

    inputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": utils.halo,
        }
    }

    halo_update_varname = "height_on_interface_levels"


class TranslateMPPUpdateDomains(TranslateHaloUpdate):

    inputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": utils.halo,
        }
    }

    halo_update_varname = "z_wind_as_tendency_of_pressure"


class TranslateHaloVectorUpdate(ParallelTranslate):

    inputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    outputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    def __init__(self, rank_grids):
        super(TranslateHaloVectorUpdate, self).__init__(rank_grids)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"starting on {communicator.rank}")
            communicator.start_vector_halo_update(
                state["x_wind_on_c_grid"],
                state["y_wind_on_c_grid"],
                n_points=utils.halo,
            )
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"finishing on {communicator.rank}")
            communicator.finish_vector_halo_update(
                state["x_wind_on_c_grid"],
                state["y_wind_on_c_grid"],
                n_points=utils.halo,
            )
        return self.outputs_list_from_state_list(state_list)
'''
