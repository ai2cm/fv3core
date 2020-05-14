from typing import List
import copy
from .translate import TranslateFortranData2Py, read_serialized_data
import fv3util
from fv3.utils import gt4py_utils as utils
import fv3


class ParallelTranslate:

    inputs = {}
    outputs = {}

    def __init__(self, rank_grids):
        self._base = TranslateFortranData2Py(rank_grids[0])
        self._base.in_vars = {
            "data_vars": {name: {} for name in self.inputs},
            "parameters": {},
        }
        self.max_error = self._base.max_error
        self._rank_grids = rank_grids  # one for each rank

    def state_list_from_inputs_list(self, inputs_list: List[list]) -> list:
        state_list = []
        for inputs, grid in zip(inputs_list, self._rank_grids):
            state_list.append(self.state_from_inputs(inputs, grid=grid))
        return state_list

    def state_from_inputs(self, inputs: dict, grid=None) -> dict:
        if grid is None:
            grid = self.grid
        inputs = copy.copy(inputs)  # don't want to modify the dict we were passed
        self._base.make_storage_data_input_vars(inputs)
        state = {}
        for name, properties in self.inputs.items():
            if len(properties["dims"]) == 3:
                state[properties["name"]] = grid.quantity_factory.empty(
                    properties["dims"],
                    properties["units"],
                    dtype=inputs[name].dtype
                )
                state[properties["name"]].data[:] = inputs[name]
            elif len(properties["dims"]) == 0:
                state[properties["name"]] = inputs[name]
            else:
                raise NotImplementedError(
                    "only 0-d parameters or 3-d variables are currently supported"
                )
        return state

    def outputs_list_from_state_list(self, state_list):
        outputs_list = []
        for state in state_list:
            outputs_list.append(self.outputs_from_state(state))
        return outputs_list

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in self.inputs.keys():
            input_data[varname] = read_serialized_data(serializer, savepoint, varname)
        return input_data

    def outputs_from_state(self, state: dict):
        return_dict = {}
        for name, properties in self.outputs.items():
            standard_name = properties["name"]
            output_slice = _serialize_slice(state[standard_name], utils.halo)
            return_dict[name] = state[standard_name].data[output_slice]
        return return_dict

    @property
    def layout(self):
        return fv3._config.namelist["layout"]

    @property
    def grid(self):
        return self._rank_grids[0]

    @property
    def rank_grids(self):
        return self._rank_grids

    def compute_sequential(self, inputs_list, communicator_list):
        """Compute the outputs while iterating over a set of communicator objects sequentially"""
        raise NotImplementedError()

    def compute_parallel(self, inputs, communicator):
        """Compute the outputs using one communicator operating in parallel"""
        self.compute_sequential(self, [inputs], [communicator])


def _serialize_slice(quantity, n_halo):
    slice_list = []
    for dim, origin, extent in zip(quantity.dims, quantity.origin, quantity.extent):
        if dim in fv3util.HORIZONTAL_DIMS:
            halo = n_halo
        else:
            halo = 0
        slice_list.append(slice(origin - halo, origin + extent + halo))
    return tuple(slice_list)
