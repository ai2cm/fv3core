import dace
import numpy as np

from fv3core.decorators import computepath_function, computepath_method
from fv3gfs.util import Quantity, QuantityHaloSpec, constants, CubedSphereCommunicator
from fv3gfs.util.halo_data_transformer import HaloExchangeSpec


MPI_Request = dace.opaque("MPI_Request")


def dace_inhibitor(f):
    return f


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


@computepath_function
def rot_vector_flatten(
    data_x,
    data_y,
    ncr: dace.constant,
    x_dim: dace.constant,
    y_dim: dace.constant,
    n_horizontal: dace.constant,
    x_in_x: dace.constant,
    y_in_y: dace.constant,
):
    x_rotated = rot_scalar_flatten(
        data_x, ncr, x_dim, y_dim, n_horizontal, x_in_x, y_in_y
    )
    y_rotated = rot_scalar_flatten(
        data_y, ncr, x_dim, y_dim, n_horizontal, x_in_x, y_in_y
    )

    if ncr == 0:
        return x_rotated, y_rotated
    if ncr == 1:
        x_final = np.empty(y_rotated.shape, dtype=y_rotated.dtype)
        y_final = np.empty(x_rotated.shape, dtype=x_rotated.dtype)
        x_final[:] = y_rotated[:]
        y_final[:] = -x_rotated[:]
        return x_final, y_final
        # x_rotated, y_rotated = y_rotated, -x_rotated
    elif ncr == 2:
        x_final_2 = np.empty(x_rotated.shape, dtype=x_rotated.dtype)
        y_final_2 = np.empty(y_rotated.shape, dtype=y_rotated.dtype)
        x_final_2[:] = -x_rotated[:]
        y_final_2[:] = -y_rotated[:]
        return x_final_2, y_final_2
        # x_rotated, y_rotated = -x_rotated, -y_rotated
    elif ncr == 3:
        x_final_3 = np.empty(y_rotated.shape, dtype=y_rotated.dtype)
        y_final_3 = np.empty(x_rotated.shape, dtype=x_rotated.dtype)
        x_final_3[:] = -y_rotated[:]
        y_final_3[:] = x_rotated[:]
        return x_final_3, y_final_3
        # x_rotated, y_rotated = -y_rotated, x_rotated


class DaceHaloUpdater:
    def __init__(
        self,
        quantity_x: Quantity,
        quantity_y: Quantity,
        comm: CubedSphereCommunicator,
        grid,
        interface=False,
        original=True,
    ):
        self._comm = comm
        self.original_updater = None
        if original:
            self.__init_original_updater(quantity_x, quantity_y, comm, grid, interface)
        else:
            self.__init_dace_halos(quantity_x, quantity_y, comm, grid, interface)

    def __init_original_updater(
        self,
        quantity_x: Quantity,
        quantity_y: Quantity,
        comm: CubedSphereCommunicator,
        grid,
        interface,
    ):
        # store quantity
        self.quantity_x = quantity_x
        self.quantity_y = quantity_y

        spec_x = QuantityHaloSpec(
            n_points=grid.halo,
            shape=quantity_x.data.shape,
            strides=quantity_x.data.strides,
            itemsize=quantity_x.data.itemsize,
            origin=quantity_x.origin,
            extent=quantity_x.extent,
            dims=quantity_x.dims,
            numpy_module=quantity_x.np,
            dtype=quantity_x.metadata.dtype,
        )

        if quantity_y is None:
            self.original_updater = comm.get_scalar_halo_updater([spec_x])
        else:
            spec_y = QuantityHaloSpec(
                n_points=grid.halo,
                shape=quantity_y.data.shape,
                strides=quantity_y.data.strides,
                itemsize=quantity_y.data.itemsize,
                origin=quantity_y.origin,
                extent=quantity_y.extent,
                dims=quantity_y.dims,
                numpy_module=quantity_y.np,
                dtype=quantity_y.metadata.dtype,
            )
            self.original_updater = comm.get_vector_halo_updater([spec_x], [spec_y])

    def __init_dace_halos(
        self,
        quantity_x: Quantity,
        quantity_y: Quantity,
        comm,
        grid,
        interface,
    ):
        # store quantity
        self.quantity_x = quantity_x
        self.quantity_y = quantity_y

        # rot config on quantity zero
        rot_config = get_rot_config(quantity_x.dims)
        self.x_dim = rot_config["x_dim"]
        self.y_dim = rot_config["y_dim"]
        self.n_horizontal = rot_config["n_horizontal"]
        self.x_in_x = rot_config["x_in_x"]
        self.y_in_y = rot_config["y_in_y"]

        if interface:
            self._init_vector_interface(self.quantity_x, self.quantity_y, comm, grid)
        else:
            # intermediates
            spec = QuantityHaloSpec(
                n_points=grid.halo,
                shape=quantity_x.data.shape,
                strides=quantity_x.data.strides,
                itemsize=quantity_x.data.itemsize,
                origin=quantity_x.origin,
                extent=quantity_x.extent,
                dims=quantity_x.dims,
                numpy_module=quantity_x.np,
                dtype=quantity_x.metadata.dtype,
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

            # attributes to use in dace programs
            self.rank_0, self.rank_1, self.rank_2, self.rank_3 = exchange_ranks
            self.send_slices = {
                rank: quantity_x.data[exchange_specs[rank].pack_slices]
                for rank in exchange_ranks
            }
            self.receive_slices = {
                rank: quantity_x.data[exchange_specs[rank].unpack_slices]
                for rank in exchange_ranks
            }
            self.receive_buffers = {
                rank: np.zeros((recv_slice.size,))
                for rank, recv_slice in self.receive_slices.items()
            }
            self.n_rotations = {
                rank: int(-exchange_specs[rank].pack_clockwise_rotation % 4)
                for rank in exchange_ranks
            }

            if quantity_y is not None:
                spec = QuantityHaloSpec(
                    n_points=grid.halo,
                    shape=quantity_y.data.shape,
                    strides=quantity_y.data.strides,
                    itemsize=quantity_y.data.itemsize,
                    origin=quantity_y.origin,
                    extent=quantity_y.extent,
                    dims=quantity_y.dims,
                    numpy_module=quantity_y.np,
                    dtype=quantity_y.metadata.dtype,
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
                self.send_slices_y = {
                    rank: quantity_y.data[exchange_specs[rank].pack_slices]
                    for rank in exchange_ranks
                }
                self.receive_slices_y = {
                    rank: quantity_y.data[exchange_specs[rank].unpack_slices]
                    for rank in exchange_ranks
                }
                self.receive_buffers_y = {
                    rank: np.zeros((recv_slice.size,))
                    for rank, recv_slice in self.receive_slices_y.items()
                }

    def _init_vector_interface(self, quantity_x, quantity_y, comm, grid):
        # Send
        self.south_boundary = comm.boundaries[constants.SOUTH]
        self.west_boundary = comm.boundaries[constants.WEST]
        self.south_data = quantity_x.view.southwest.sel(
            **{
                constants.Y_INTERFACE_DIM: 0,
                constants.X_DIM: slice(
                    0, quantity_x.extent[quantity_x.dims.index(constants.X_DIM)]
                ),
            }
        )
        self.south_config = get_rot_config([constants.X_DIM])
        self.west_data = quantity_y.view.southwest.sel(
            **{
                constants.X_INTERFACE_DIM: 0,
                constants.Y_DIM: slice(
                    0, quantity_y.extent[quantity_y.dims.index(constants.Y_DIM)]
                ),
            }
        )
        self.west_config = get_rot_config([constants.Y_DIM])
        # Recv
        self.north_rank = comm.boundaries[constants.NORTH].to_rank
        self.east_rank = comm.boundaries[constants.EAST].to_rank
        self.north_data = quantity_x.view.northwest.sel(
            **{
                constants.Y_INTERFACE_DIM: -1,
                constants.X_DIM: slice(
                    0, quantity_x.extent[quantity_x.dims.index(constants.X_DIM)]
                ),
            }
        )
        self.east_data = quantity_y.view.southeast.sel(
            **{
                constants.X_INTERFACE_DIM: -1,
                constants.Y_DIM: slice(
                    0, quantity_y.extent[quantity_y.dims.index(constants.Y_DIM)]
                ),
            }
        )
        self.east_buffer = np.zeros(self.east_data.shape).flatten()
        self.north_buffer = np.zeros(self.north_data.shape).flatten()

    @dace_inhibitor
    def original_start_halo_update(self):
        if self.quantity_y is None:
            self.original_updater.start([self.quantity_x])
        else:
            self.original_updater.start([self.quantity_x], [self.quantity_y])

    @dace_inhibitor
    def original_finish_halo_update(self):
        self.original_updater.wait()

    @dace_inhibitor
    def original_vector_interface_update(self):
        self._comm.synchronize_vector_interfaces(self.quantity_x, self.quantity_y)

    @computepath_method(use_dace=True)
    def start_halo_update(self):
        if self.original_updater is not None:
            self.original_start_halo_update()
        else:
            # Scalar
            if self.quantity_y is None:
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
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_0], self.rank_0, 0, req[1]
                )

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
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_1], self.rank_1, 0, req[3]
                )

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
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_2], self.rank_2, 0, req[5]
                )

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
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_3], self.rank_3, 0, req[7]
                )

                dace.comm.Waitall(req)

            else:  # Vector
                req = np.empty((16,), dtype=MPI_Request)
                x_0, y_0 = rot_vector_flatten(
                    self.send_slices[self.rank_0],
                    self.send_slices_y[self.rank_0],
                    self.n_rotations[self.rank_0],
                    self.x_dim,
                    self.y_dim,
                    self.n_horizontal,
                    self.x_in_x,
                    self.y_in_y,
                )
                dace.comm.Isend(x_0, self.rank_0, 0, req[0])
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_0], self.rank_0, 0, req[1]
                )
                dace.comm.Isend(y_0, self.rank_0, 0, req[2])
                dace.comm.Irecv(
                    self.receive_buffers_y[self.rank_0], self.rank_0, 0, req[3]
                )

                x_1, y_1 = rot_vector_flatten(
                    self.send_slices[self.rank_1],
                    self.send_slices_y[self.rank_1],
                    self.n_rotations[self.rank_1],
                    self.x_dim,
                    self.y_dim,
                    self.n_horizontal,
                    self.x_in_x,
                    self.y_in_y,
                )
                dace.comm.Isend(x_1, self.rank_1, 0, req[4])
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_1], self.rank_1, 0, req[5]
                )
                dace.comm.Isend(y_1, self.rank_1, 0, req[6])
                dace.comm.Irecv(
                    self.receive_buffers_y[self.rank_1], self.rank_1, 0, req[7]
                )

                x_2, y_2 = rot_vector_flatten(
                    self.send_slices[self.rank_2],
                    self.send_slices_y[self.rank_2],
                    self.n_rotations[self.rank_2],
                    self.x_dim,
                    self.y_dim,
                    self.n_horizontal,
                    self.x_in_x,
                    self.y_in_y,
                )
                dace.comm.Isend(x_2, self.rank_2, 0, req[8])
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_2], self.rank_2, 0, req[9]
                )
                dace.comm.Isend(y_2, self.rank_2, 0, req[10])
                dace.comm.Irecv(
                    self.receive_buffers_y[self.rank_2], self.rank_2, 0, req[11]
                )

                x_3, y_3 = rot_vector_flatten(
                    self.send_slices[self.rank_3],
                    self.send_slices_y[self.rank_3],
                    self.n_rotations[self.rank_3],
                    self.x_dim,
                    self.y_dim,
                    self.n_horizontal,
                    self.x_in_x,
                    self.y_in_y,
                )
                dace.comm.Isend(x_3, self.rank_3, 0, req[12])
                dace.comm.Irecv(
                    self.receive_buffers[self.rank_3], self.rank_3, 0, req[13]
                )
                dace.comm.Isend(y_3, self.rank_3, 0, req[14])
                dace.comm.Irecv(
                    self.receive_buffers_y[self.rank_3], self.rank_3, 0, req[15]
                )

                dace.comm.Waitall(req)

    @computepath_method(use_dace=True)
    def finish_halo_update(self):
        if self.original_updater is not None:
            self.original_finish_halo_update()
        else:
            self.receive_slices[self.rank_0][:] = np.reshape(
                self.receive_buffers[self.rank_0],
                self.receive_slices[self.rank_0].shape,
            )
            self.receive_slices[self.rank_1][:] = np.reshape(
                self.receive_buffers[self.rank_1],
                self.receive_slices[self.rank_1].shape,
            )
            self.receive_slices[self.rank_2][:] = np.reshape(
                self.receive_buffers[self.rank_2],
                self.receive_slices[self.rank_2].shape,
            )
            self.receive_slices[self.rank_3][:] = np.reshape(
                self.receive_buffers[self.rank_3],
                self.receive_slices[self.rank_3].shape,
            )
            if self.quantity_y is not None:
                self.receive_slices_y[self.rank_0][:] = np.reshape(
                    self.receive_buffers_y[self.rank_0],
                    self.receive_slices_y[self.rank_0].shape,
                )
                self.receive_slices_y[self.rank_1][:] = np.reshape(
                    self.receive_buffers_y[self.rank_1],
                    self.receive_slices_y[self.rank_1].shape,
                )
                self.receive_slices_y[self.rank_2][:] = np.reshape(
                    self.receive_buffers_y[self.rank_2],
                    self.receive_slices_y[self.rank_2].shape,
                )
                self.receive_slices_y[self.rank_3][:] = np.reshape(
                    self.receive_buffers_y[self.rank_3],
                    self.receive_slices_y[self.rank_3].shape,
                )

    @computepath_method(use_dace=True)
    def do_halo_update(self):
        self.start_halo_update()
        self.finish_halo_update()

    @computepath_method(use_dace=True)
    def do_halo_vector_update(self):
        self.start_halo_update()
        self.finish_halo_update()

    def _dace_vector_interface(self):
        req = np.empty((4,), dtype=MPI_Request)
        south_data_rotated = rot_scalar_flatten(
            self.south_data,
            -self.south_boundary.n_clockwise_rotations,
            self.south_config["x_dim"],
            self.south_config["y_dim"],
            self.south_config["n_horizontal"],
            self.south_config["x_in_x"],
            self.south_config["y_in_y"],
        )
        if self.south_boundary.n_clockwise_rotations in (3, 2):
            south_data_rotated *= -1
        dace.comm.Isend(south_data_rotated, self.south_boundary.to_rank, 0, req[0])
        dace.comm.Irecv(self.north_buffer, self.north_rank, 0, req[1])

        west_data_rotated = rot_scalar_flatten(
            self.west_data,
            -self.west_boundary.n_clockwise_rotations,
            self.west_config["x_dim"],
            self.west_config["y_dim"],
            self.west_config["n_horizontal"],
            self.west_config["x_in_x"],
            self.west_config["y_in_y"],
        )
        if self.west_boundary.n_clockwise_rotations in (1, 2):
            west_data_rotated *= -1
        dace.comm.Isend(west_data_rotated, self.west_boundary.to_rank, 0, req[2])
        dace.comm.Irecv(self.east_buffer, self.east_rank, 0, req[3])

        dace.comm.Waitall(req)

        self.east_data[:] = np.reshape(self.east_buffer, self.east_data.shape)
        self.north_data[:] = np.reshape(self.north_buffer, self.north_data.shape)

    @computepath_method(use_dace=True)
    def do_halo_vector_interface_update(self, original=True):
        if original:
            self.original_vector_interface_update()
        else:
            self._dace_vector_interface()
