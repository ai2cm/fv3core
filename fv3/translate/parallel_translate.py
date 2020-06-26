from typing import List
import copy
from .translate import TranslateFortranData2Py, read_serialized_data
import fv3util
from fv3.utils import gt4py_utils as utils
import fv3
import pytest
from types import SimpleNamespace


def ensure_3d_dims(dims_in):
    dims_out = [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM]
    for dim in dims_in:
        for i, dim_set in enumerate([fv3util.X_DIMS, fv3util.Y_DIMS, fv3util.Z_DIMS]):
            if dim in dim_set:
                dims_out[i] = dim
                break
        else:
            raise ValueError(f"dimension {dim} is not an x/y/z dimension")
    return dims_out


class ParallelTranslate:

    max_error = TranslateFortranData2Py.max_error

    inputs = {}
    outputs = {}

    def __init__(self, rank_grids):
        if not hasattr(rank_grids, "__getitem__"):
            raise TypeError(
                "rank_grids should be a sequence of grids, one for each rank, "
                f"is {self.__class__} being properly called as a parallel test?"
            )
        self._base = TranslateFortranData2Py(rank_grids[0])
        self._base.in_vars = {
            "data_vars": {name: {} for name in self.inputs},
            "parameters": [],
        }
        self.max_error = self._base.max_error
        self._rank_grids = rank_grids
        self.ignore_near_zero_errors = {}

    def state_list_from_inputs_list(self, inputs_list: List[list]) -> list:
        state_list = []
        for inputs in inputs_list:
            state_list.append(self.state_from_inputs(inputs))
        return state_list

    def state_from_inputs(self, inputs: dict, grid=None) -> dict:
        if grid is None:
            grid = self.grid
        state = copy.copy(inputs)
        self._base.make_storage_data_input_vars(state)
        for name, properties in self.inputs.items():
            if "name" not in properties:
                properties["name"] = name
            input_data = state[name]
            if len(properties["dims"]) > 0:
                # self._base will always make a 3D array
                dims = ensure_3d_dims(properties["dims"])
                state[properties["name"]] = fv3util.Quantity(
                    input_data,
                    dims,
                    properties["units"],
                    origin=grid.sizer.get_origin(dims),
                    extent=grid.sizer.get_extent(dims),
                )
            else:
                state[properties["name"]] = input_data
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
        if len(self.outputs) == 0:
            return {}
        outputs = {}
        storages = {}
        for name in self.outputs:
            if isinstance(state[name], fv3util.Quantity):
                storages[name] = state[name].storage
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[name]  # assume it's a storage
            else:
                outputs[name] = state[name]  # scalar
        outputs.update(self._base.slice_output(storages))
        return outputs
        #     name: state[name].storage for name, q in self.outputs if isinstance(state[name], fv3util.Quantity)
        # }
        # for name, properties in self.outputs.items():
        #     standard_name = properties["name"]
        #     output_slice = _serialize_slice(
        #         state[standard_name], properties.get("n_halo", utils.halo), real_dims=properties["dims"]
        #     )
        #     return_dict[name] = state[standard_name].data[output_slice]
        # return return_dict

    @property
    def rank_grids(self):
        return self._rank_grids

    @property
    def grid(self):
        return self._rank_grids[0]

    @property
    def layout(self):
        return fv3._config.namelist["layout"]

    def compute_sequential(self, inputs_list, communicator_list):
        """Compute the outputs while iterating over a set of communicator objects sequentially"""
        raise NotImplementedError()

    def compute_parallel(self, inputs, communicator):
        """Compute the outputs using one communicator operating in parallel"""
        self.compute_sequential(self, [inputs], [communicator])


def _serialize_slice(quantity, n_halo, real_dims=None):
    if real_dims is None:
        real_dims = quantity.dims
    slice_list = []
    for dim, origin, extent in zip(quantity.dims, quantity.origin, quantity.extent):
        if dim in real_dims:
            if dim in fv3util.HORIZONTAL_DIMS:
                if isinstance(n_halo, int):
                    halo = n_halo
                elif dim in fv3util.X_DIMS:
                    halo = n_halo[0]
                elif dim in fv3util.Y_DIMS:
                    halo = n_halo[1]
                else:
                    raise RuntimeError(n_halo)
            else:
                halo = 0
            slice_list.append(slice(origin - halo, origin + extent + halo))
        else:
            slice_list.append(-1)
    return tuple(slice_list)


class ParallelTranslate2Py(ParallelTranslate):
    def collect_input_data(self, serializer, savepoint):
        input_data = super().collect_input_data(serializer, savepoint)
        input_data.update(self._base.collect_input_data(serializer, savepoint))
        return input_data

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator
        inputs = self.state_from_inputs(inputs)
        result = self._base.compute_from_storage(inputs)
        quantity_result = self.outputs_from_state(inputs)
        result.update(quantity_result)
        return result

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, not running in mock-parallel"
        )


class ParallelTranslate2PyState(ParallelTranslate2Py):
    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        for name, properties in self.inputs.items():
            self.grid.quantity_dict_update(
                inputs, name, dims=properties["dims"], units=properties["units"]
            )
        statevars = SimpleNamespace(**inputs)
        state = {"state": statevars, "comm": communicator}
        self._base.compute_func(**state)
        return self._base.slice_output(vars(state["state"]))


class ParallelTranslateGrid(ParallelTranslate):
    """Translation class which only uses quantity factory for initialization, to
    support some non-standard array dimension layouts not supported by the
    TranslateFortranData2Py initializers.
    """
    
    def state_from_inputs(self, inputs: dict, grid=None) -> dict:
        if grid is None:
            grid = self.grid
        state = {}
        for name, properties in self.inputs.items():
            state_name = properties.get("name", name)
            if len(properties["dims"]) > 0:
                state[state_name] = grid.quantity_factory.empty(
                    properties["dims"],
                    properties["units"],
                    dtype=inputs[name].dtype
                )
                input_slice = _serialize_slice(state[state_name], properties.get("n_halo", utils.halo))
                state[state_name].data[input_slice] = inputs[name]
                if len(properties["dims"]) > 0:
                    state[state_name].data[input_slice] = inputs[name]
                else:
                    state[state_name].data[:] = inputs[name]
            else:
                state[state_name] = inputs[name]
        return state
