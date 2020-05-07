from .parallel_translate import ParallelTranslate,_serialize_slice
from .translate import TranslateFortranData2Py
import fv3util
from fv3.utils import gt4py_utils as utils
import logging
from mpi4py import MPI
import numpy as np
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
        state = self.state_from_inputs(inputs)
        name = self.halo_update_varname 
        datapoint = 70.6421745627577
        #print('rank',rank_communicator.rank,'shape',state[name].data.shape, 'data', state[name].data[0, 4, 0])
        before = np.argwhere(state[name].data == datapoint)
        if len(before) > 0:
            print('\nBEFORE HALO UPDATE rank', rank_communicator.rank, 'at', before)
        
        rank_communicator.start_halo_update(state[name], n_points=utils.halo)
       
        rank_communicator.finish_halo_update(state[name], n_points=utils.halo)
        found = np.argwhere(state[name].data == datapoint)
        if len(found) > 0:
            print('AFTER FOUND at rank',rank_communicator.rank, 'at', found)
        #print('after, rank',rank_communicator.rank, state[name].data[0, 4, 0])
       
        return self.outputs_from_state(state)
   

   
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
