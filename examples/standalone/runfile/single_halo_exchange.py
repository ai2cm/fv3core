from types import SimpleNamespace

import click
import dace
import numpy as np

import serialbox
from fv3gfs.util import CubedSphereCommunicator, CubedSpherePartitioner, TilePartitioner
from fv3gfs.util import QuantityHaloSpec
from fv3gfs.util import constants
from fv3gfs.util.halo_data_transformer import HaloExchangeSpec

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
from fv3core.decorators import computepath_function, computepath_method
from fv3core.utils.mpi import MPI


MPI_Request = dace.opaque("MPI_Request")


def set_up_namelist(data_directory: str) -> None:
    spec.set_namelist(data_directory + "/input.nml")


def initialize_serializer(data_directory: str, rank: int = 0) -> serialbox.Serializer:
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )


def read_grid(
    serializer: serialbox.Serializer, rank: int = 0
) -> fv3core.testing.TranslateGrid:
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    return fv3core.testing.TranslateGrid(grid_data, rank).python_grid()


def initialize_fv3core(backend: str) -> None:
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)
    global_config.set_do_halo_exchange(True)


def read_input_data(grid, serializer):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(grid, input_data):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return statevars


def do_halo_update(state: dace.constant, comm: dace.constant, grid: dace.constant):
    halo_request = comm.start_halo_update(
        state.__getattribute__("q_con_quantity"), n_points=grid.halo
    )
    halo_request.wait()
    return state.q_con_quantity


def get_rot_config(dims):
    x_dim, y_dim = 9999, 9999
    x_x = False
    y_y = False
    for i, dim in enumerate(dims):
        if dim in constants.X_DIMS:
            x_dim = i
            x_x = True
        elif dim in constants.Y_DIMS:
            y_dim = i
            y_y = True

    n_horizontal = sum(int(dim in constants.HORIZONTAL_DIMS) for dim in dims)
    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "x_in_x": x_x,
        "y_in_y": y_y,
        "n_horizontal": n_horizontal,
    }


@computepath_function
def rot_scalar_flatten(
    data,
    ncr: dace.int32,
    x_dim: dace.constant,
    y_dim: dace.constant,
    n_horizontal: dace.constant,
    x_in_x: dace.constant,
    y_in_y: dace.constant,
):
    """
    DaCe equivalent of rotate_scalar_data followed by .flatten().

    Changes to the API:

    * ncr has to be positive, because modulo works differently in pure python
    * the rest of the inputs can be obtained through `get_rot_config` by passing
        the `dims` argument to it. Unfortunately they can not be passed as **kwargs.
    """
    n_clockwise_rotations = ncr % 4
    result = np.zeros((data.size,), dtype=data.dtype)
    tmp_00 = np.zeros(data.shape, dtype=data.dtype)

    if n_clockwise_rotations == 0:
        tmp_00[:] = data[:]
        result[:] = tmp_00.flatten()

    elif n_clockwise_rotations == 1 or n_clockwise_rotations == 3:
        if x_in_x & y_in_y:
            if n_clockwise_rotations == 1:
                tmp_0 = np.rot90(data, axes=(y_dim, x_dim))
                result[:] = tmp_0.flatten()
            if n_clockwise_rotations == 3:
                tmp_1 = np.rot90(data, axes=(x_dim, y_dim))
                result[:] = tmp_1.flatten()

        elif x_in_x:
            if n_clockwise_rotations == 1:
                tmp_2 = np.flip(data, axis=x_dim)
                result[:] = tmp_2.flatten()

        elif y_in_y:
            if n_clockwise_rotations == 3:
                tmp_3 = np.flip(data, axis=y_dim)
                result[:] = tmp_3.flatten()

    elif n_clockwise_rotations == 2:
        if n_horizontal == 1:
            tmp_4 = np.flip(data, axis=0)
            result[:] = tmp_4.flatten()
        elif n_horizontal == 2:
            tmp_5 = np.flip(data, axis=(0, 1))
            result[:] = tmp_5.flatten()
        elif n_horizontal == 3:
            tmp_6 = np.flip(data, axis=(0, 1, 2))
            result[:] = tmp_6.flatten()
    else:
        result[:] = data.flatten()
    return result


class DaceHaloUpdater:
    def __init__(self, quantity, comm, grid):
        # store quantity
        self.quantity = quantity

        # intermediates
        spec = QuantityHaloSpec(
            n_points=grid.halo,
            shape=quantity.data.shape,
            strides=quantity.data.strides,
            itemsize=quantity.data.itemsize,
            origin=quantity.origin,
            extent=quantity.extent,
            dims=quantity.dims,
            numpy_module=quantity.np,
            dtype=quantity.metadata.dtype,
        )
        exchange_specs = {
            boundary.to_rank: HaloExchangeSpec(
                spec,
                boundary.send_slice(spec),
                boundary.n_clockwise_rotations,
                boundary.recv_slice(spec),
            )
            for boundary in comm.boundaries.values()
        }
        exchange_ranks = [boundary.to_rank for boundary in comm.boundaries.values()]
        rot_config = get_rot_config(quantity.dims)

        # attributes to use in dace programs
        self.rank_0, self.rank_1, self.rank_2, self.rank_3 = exchange_ranks
        self.send_slices = {
            rank: quantity.data[exchange_specs[rank].pack_slices]
            for rank in exchange_ranks
        }
        self.receive_slices = {
            rank: quantity.data[exchange_specs[rank].unpack_slices]
            for rank in exchange_ranks
        }
        self.receive_buffers = {
            rank: np.zeros((recv_slice.size,))
            for rank, recv_slice in self.receive_slices.items()
        }
        self.n_rotations = {
            rank: -exchange_specs[rank].pack_clockwise_rotation % 4
            for rank in exchange_ranks
        }
        self.x_dim = rot_config["x_dim"]
        self.y_dim = rot_config["y_dim"]
        self.n_horizontal = rot_config["n_horizontal"]
        self.x_in_x = rot_config["x_in_x"]
        self.y_in_y = rot_config["y_in_y"]

    @computepath_method(use_dace=True)
    def start_halo_exchange(self):
        req = np.empty((8,), dtype=MPI_Request)

        dace.comm.Isend(
            rot_scalar_flatten(
                self.send_slices[self.rank_0],
                self.n_rotations[self.rank_0],
                self.x_dim,
                self.y_dim,
                self.n_horizontal,
                self.x_in_x,
                self.y_in_y,
            ),
            self.rank_0,
            0,
            req[0],
        )
        dace.comm.Irecv(self.receive_buffers[self.rank_0], self.rank_0, 0, req[1])

        dace.comm.Isend(
            rot_scalar_flatten(
                self.send_slices[self.rank_1],
                self.n_rotations[self.rank_1],
                self.x_dim,
                self.y_dim,
                self.n_horizontal,
                self.x_in_x,
                self.y_in_y,
            ),
            self.rank_1,
            0,
            req[2],
        )
        dace.comm.Irecv(self.receive_buffers[self.rank_1], self.rank_1, 0, req[3])

        dace.comm.Isend(
            rot_scalar_flatten(
                self.send_slices[self.rank_2],
                self.n_rotations[self.rank_2],
                self.x_dim,
                self.y_dim,
                self.n_horizontal,
                self.x_in_x,
                self.y_in_y,
            ),
            self.rank_2,
            0,
            req[4],
        )
        dace.comm.Irecv(self.receive_buffers[self.rank_2], self.rank_2, 0, req[5])

        dace.comm.Isend(
            rot_scalar_flatten(
                self.send_slices[self.rank_3],
                self.n_rotations[self.rank_3],
                self.x_dim,
                self.y_dim,
                self.n_horizontal,
                self.x_in_x,
                self.y_in_y,
            ),
            self.rank_3,
            0,
            req[6],
        )
        dace.comm.Irecv(self.receive_buffers[self.rank_3], self.rank_3, 0, req[7])

        dace.comm.Waitall(req)

    @computepath_method(use_dace=True)
    def finish_halo_exchange(self):
        self.receive_slices[self.rank_0][:] = np.reshape(
            self.receive_buffers[self.rank_0], self.receive_slices[self.rank_0].shape
        )
        self.receive_slices[self.rank_1][:] = np.reshape(
            self.receive_buffers[self.rank_1], self.receive_slices[self.rank_1].shape
        )
        self.receive_slices[self.rank_2][:] = np.reshape(
            self.receive_buffers[self.rank_2], self.receive_slices[self.rank_2].shape
        )
        self.receive_slices[self.rank_3][:] = np.reshape(
            self.receive_buffers[self.rank_3], self.receive_slices[self.rank_3].shape
        )


    @computepath_method(use_dace=True)
    def do_halo_exchange(self):
        self.start_halo_exchange()
        self.finish_halo_exchange()


def run(data_directory, backend):
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)
    state = get_state_from_input(grid, input_data)

    input_data2 = read_input_data(grid, serializer)
    state2 = get_state_from_input(grid, input_data2)
    q_con2 = state2.q_con_quantity

    layout = spec.namelist.layout

    comm = CubedSphereCommunicator(
        comm=MPI.COMM_WORLD, partitioner=CubedSpherePartitioner(TilePartitioner(layout))
    )

    q_con_hex = do_halo_update(state, comm, grid)

    updater = DaceHaloUpdater(q_con2, comm, grid)
    updater.do_halo_exchange()
    # ~ updater.start_halo_exchange()
    # ~ updater.finish_halo_exchange()

    np.testing.assert_allclose(
        np.asarray(q_con2.storage), np.asarray(q_con_hex.storage)
    )

    return state


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
def driver(
    data_directory: str,
    backend: str,
):
    state = run(data_directory, backend)
    if MPI.COMM_WORLD.Get_rank() == 0:
        click.echo("Done")


if __name__ == "__main__":
    driver()
