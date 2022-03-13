import dace
import numpy as np

from fv3core.utils.dace.computepath import computepath_function, computepath_method
from fv3gfs.util import Quantity, QuantityHaloSpec, constants, CubedSphereCommunicator
from fv3gfs.util.halo_data_transformer import HaloExchangeSpec
from mpi4py import MPI

MPI_Request = dace.opaque("MPI_Request")


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

    if n_clockwise_rotations == 0:
        tmp_0 = np.zeros(data.shape, dtype=data.dtype)
        tmp_0[:] = data[:]
        result[:] = tmp_0.flatten()

    elif n_clockwise_rotations == 1 or n_clockwise_rotations == 3:
        if x_in_x & y_in_y:
            if n_clockwise_rotations == 1:
                tmp_1 = np.rot90(data, axes=(y_dim, x_dim))
                result[:] = tmp_1.flatten()
            if n_clockwise_rotations == 3:
                tmp_3 = np.rot90(data, axes=(x_dim, y_dim))
                result[:] = tmp_3.flatten()

        elif x_in_x:
            if n_clockwise_rotations == 1:
                tmp_1x = np.empty_like(data)
                tmp_1x = np.flip(data, axis=x_dim)
                result[:] = tmp_1x.flatten()

        elif y_in_y:
            if n_clockwise_rotations == 3:
                tmp_3y = np.empty_like(data)
                tmp_3y = np.flip(data, axis=y_dim)
                result[:] = tmp_3y.flatten()

        else:
            raise RuntimeError(f"Unexpected rotation {n_clockwise_rotations}")
    elif n_clockwise_rotations == 2:
        if n_horizontal == 1:
            tmp_4 = np.empty_like(data)
            tmp_4[:] = np.flip(data, axis=0)
            result[:] = tmp_4.flatten()

        elif n_horizontal == 2:
            tmp_5 = np.empty_like(data)
            tmp_5[:] = np.flip(data, axis=(0, 1))
            result[:] = tmp_5.flatten()

        elif n_horizontal == 3:
            tmp_6 = np.empty_like(data)
            tmp_6[:] = np.flip(data, axis=(0, 1, 2))
            result[:] = tmp_6.flatten()
        else:
            raise RuntimeError(f"Unexpected rotation {n_clockwise_rotations}")

    else:
        raise RuntimeError(f"Unexpected rotation {n_clockwise_rotations}")
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
    ):
        self._comm = comm
        self.original_updater = None
        self.__init_dace_halos(quantity_x, quantity_y, comm, grid, interface)

    def __init_dace_halos(
        self,
        quantity_x: Quantity,
        quantity_y: Quantity,
        comm,
        grid,
        interface,
    ):
        # Cache the current np-like module used
        self.np = quantity_x.np

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

        # pre allocated request
        self.is_scalar_exchanged = quantity_y is None

        if interface:
            self._init_vector_interface(quantity_x, quantity_y, comm, grid)
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
            # Moved to transient because of MPI + CUDA + DAINT
            # self.receive_buffers = {
            #     rank: np.empty((recv_slice.size,), dtype=self.send_slices_0.dtype)
            #     for rank, recv_slice in self.receive_slices.items()
            # }
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
                # Moved to transient because of MPI + CUDA + DAINT
                # self.receive_buffers_y = {
                #     rank: np.zeros((recv_slice.size,))
                #     for rank, recv_slice in self.receive_slices_y.items()
                # }

    def _init_vector_interface(self, quantity_x, quantity_y, comm, grid):
        self.rank = comm.comm.Get_rank()
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
        self.east_buffer = self.np.zeros(self.east_data.shape).flatten()
        self.north_buffer = self.np.zeros(self.north_data.shape).flatten()

    @computepath_method(use_dace=True)
    def start_halo_update(self):
        if self.is_scalar_exchanged:
            req = np.empty((8,), dtype=MPI_Request)
        else:
            req = np.empty((16,), dtype=MPI_Request)
        # Scalar
        if self.is_scalar_exchanged:
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
            recv_buffer_0_x = np.empty(
                (self.send_slices[self.rank_0].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_0_x, self.rank_0, 0, req[1])
            # dace.comm.Irecv(self.receive_buffers[self.rank_0], self.rank_0, 0, req[1])

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
            recv_buffer_1_x = np.empty(
                (self.send_slices[self.rank_1].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_1_x, self.rank_1, 0, req[3])
            # dace.comm.Irecv(self.receive_buffers[self.rank_1], self.rank_1, 0, req[3])

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
            recv_buffer_2_x = np.empty(
                (self.send_slices[self.rank_2].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_2_x, self.rank_2, 0, req[5])
            # dace.comm.Irecv(self.receive_buffers[self.rank_2], self.rank_2, 0, req[5])

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
            recv_buffer_3_x = np.empty(
                (self.send_slices[self.rank_3].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_3_x, self.rank_3, 0, req[7])
            # dace.comm.Irecv(self.receive_buffers[self.rank_3], self.rank_3, 0, req[7])

        else:  # Vector
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
            recv_buffer_0_x = np.empty(
                (self.send_slices[self.rank_0].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_0_x, self.rank_0, 0, req[1])
            dace.comm.Isend(y_0, self.rank_0, 0, req[2])
            recv_buffer_0_y = np.empty(
                (self.send_slices_y[self.rank_0].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_0_y, self.rank_0, 0, req[3])

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
            recv_buffer_1_x = np.empty(
                (self.send_slices[self.rank_1].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_1_x, self.rank_1, 0, req[5])
            dace.comm.Isend(y_1, self.rank_1, 0, req[6])
            recv_buffer_1_y = np.empty(
                (self.send_slices_y[self.rank_1].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_1_y, self.rank_1, 0, req[7])

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
            recv_buffer_2_x = np.empty(
                (self.send_slices[self.rank_2].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_2_x, self.rank_2, 0, req[9])
            dace.comm.Isend(y_2, self.rank_2, 0, req[10])
            recv_buffer_2_y = np.empty(
                (self.send_slices_y[self.rank_2].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_2_y, self.rank_2, 0, req[11])

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
            recv_buffer_3_x = np.empty(
                (self.send_slices[self.rank_3].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_3_x, self.rank_3, 0, req[13])
            dace.comm.Isend(y_3, self.rank_3, 0, req[14])
            recv_buffer_3_y = np.empty(
                (self.send_slices_y[self.rank_3].size,), dtype=dace.float64
            )
            dace.comm.Irecv(recv_buffer_3_y, self.rank_3, 0, req[15])

        dace.comm.Waitall(req)

        self.receive_slices[self.rank_0][:] = np.reshape(
            recv_buffer_0_x, self.receive_slices[self.rank_0].shape
        )
        self.receive_slices[self.rank_1][:] = np.reshape(
            recv_buffer_1_x, self.receive_slices[self.rank_1].shape
        )
        self.receive_slices[self.rank_2][:] = np.reshape(
            recv_buffer_2_x, self.receive_slices[self.rank_2].shape
        )
        self.receive_slices[self.rank_3][:] = np.reshape(
            recv_buffer_3_x, self.receive_slices[self.rank_3].shape
        )
        if not self.is_scalar_exchanged:
            self.receive_slices_y[self.rank_0][:] = np.reshape(
                recv_buffer_0_y,
                self.receive_slices_y[self.rank_0].shape,
            )
            self.receive_slices_y[self.rank_1][:] = np.reshape(
                recv_buffer_1_y,
                self.receive_slices_y[self.rank_1].shape,
            )
            self.receive_slices_y[self.rank_2][:] = np.reshape(
                recv_buffer_2_y,
                self.receive_slices_y[self.rank_2].shape,
            )
            self.receive_slices_y[self.rank_3][:] = np.reshape(
                recv_buffer_3_y,
                self.receive_slices_y[self.rank_3].shape,
            )

    @computepath_method(use_dace=True)
    def finish_halo_update(self):
        # MPI + CUDA was giving crash on Daint when using outside of Dace allocated
        # receive buffers. We moved those to be transient and therefore moved the
        # copy back right after the wait all in `start_halo_update`
        pass
        # self.receive_slices[self.rank_0][:] = np.reshape(
        #     self.receive_buffers[self.rank_0], self.receive_slices[self.rank_0].shape
        # )
        # self.receive_slices[self.rank_1][:] = np.reshape(
        #     self.receive_buffers[self.rank_1], self.receive_slices[self.rank_1].shape
        # )
        # self.receive_slices[self.rank_2][:] = np.reshape(
        #     self.receive_buffers[self.rank_2], self.receive_slices[self.rank_2].shape
        # )
        # self.receive_slices[self.rank_3][:] = np.reshape(
        #     self.receive_buffers[self.rank_3], self.receive_slices[self.rank_3].shape
        # )
        # if not self.is_scalar_exchanged:
        #     self.receive_slices_y[self.rank_0][:] = np.reshape(
        #         self.receive_buffers_y[self.rank_0],
        #         self.receive_slices_y[self.rank_0].shape,
        #     )
        #     self.receive_slices_y[self.rank_1][:] = np.reshape(
        #         self.receive_buffers_y[self.rank_1],
        #         self.receive_slices_y[self.rank_1].shape,
        #     )
        #     self.receive_slices_y[self.rank_2][:] = np.reshape(
        #         self.receive_buffers_y[self.rank_2],
        #         self.receive_slices_y[self.rank_2].shape,
        #     )
        #     self.receive_slices_y[self.rank_3][:] = np.reshape(
        #         self.receive_buffers_y[self.rank_3],
        #         self.receive_slices_y[self.rank_3].shape,
        #     )

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
            south_data_rotated[:] = -1 * south_data_rotated[:]

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
            west_data_rotated[:] = -1 * west_data_rotated[:]

        dace.comm.Isend(south_data_rotated, self.south_boundary.to_rank, 1, req[0])
        dace.comm.Isend(west_data_rotated, self.west_boundary.to_rank, 1, req[2])

        dace.comm.Irecv(self.east_buffer, self.east_rank, 1, req[3])
        dace.comm.Irecv(self.north_buffer, self.north_rank, 1, req[1])

        dace.comm.Waitall(req)

        self.east_data[:] = np.reshape(self.east_buffer, self.east_data.shape)
        self.north_data[:] = np.reshape(self.north_buffer, self.north_data.shape)

    @computepath_method(use_dace=True)
    def do_halo_vector_interface_update(self):
        self._dace_vector_interface()

    @computepath_method
    def do_rot_0_cpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - Config for 0,"
            f"  ncr {self.n_rotations[self.rank_0]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_0,
            self.n_rotations[self.rank_0],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_1_cpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - Config for 1,"
            f"  ncr {self.n_rotations[self.rank_1]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_1,
            self.n_rotations[self.rank_1],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_2_cpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - Config for 2,"
            f"  ncr {self.n_rotations[self.rank_2]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_2,
            self.n_rotations[self.rank_2],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_3_cpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - Config for 3,"
            f"  ncr {self.n_rotations[self.rank_3]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_3,
            self.n_rotations[self.rank_3],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_0_gpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - GConfig for 0,"
            f"  ncr {self.n_rotations[self.rank_0]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_0,
            self.n_rotations[self.rank_0],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_1_gpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - GConfig for 1,"
            f"  ncr {self.n_rotations[self.rank_1]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_1,
            self.n_rotations[self.rank_1],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_2_gpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - GConfig for 2,"
            f"  ncr {self.n_rotations[self.rank_2]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_2,
            self.n_rotations[self.rank_2],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_rot_3_gpu(self):
        print(
            f"r{MPI.COMM_WORLD.Get_rank()} - GConfig for 3,"
            f"  ncr {self.n_rotations[self.rank_3]},"
            f"  x,y_dim {self.x_dim}, {self.y_dim},"
            f"  n_horiz {self.n_horizontal},"
            f"  xy_in_xy {self.x_in_x}, {self.y_in_y}"
        )
        return rot_scalar_flatten(
            self.send_slices_3,
            self.n_rotations[self.rank_3],
            self.x_dim,
            self.y_dim,
            self.n_horizontal,
            self.x_in_x,
            self.y_in_y,
        )

    @computepath_method
    def do_minimal_failure_cpu(self):
        result = np.zeros((self.send_slices_2.size,), dtype=self.send_slices_2.dtype)
        tmp_1 = np.rot90(self.send_slices_2, axes=(0, 1))
        result[:] = tmp_1.flatten()
        return result

    @computepath_method
    def do_minimal_failure_gpu(self):
        result = np.zeros((self.send_slices_2.size,), dtype=self.send_slices_2.dtype)
        tmp_1 = np.rot90(self.send_slices_2, axes=(0, 1))
        result[:] = tmp_1.flatten()
        return result
