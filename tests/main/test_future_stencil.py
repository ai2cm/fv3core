import os
import random as rand
import shutil
import time
from typing import Callable, Tuple

import gt4py as gt
import gt4py.storage as gt_storage
import numpy as np
import pytest
from gt4py.gtscript import PARALLEL, computation, interval
from gt4py.stencil_object import StencilObject

from fv3core.utils.future_stencil import FutureStencil, StencilTable, future_stencil
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.mpi import MPI
from fv3core.utils.typing import FloatField, IntField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add1_stencil(q: FloatField):
    with computation(PARALLEL), interval(...):
        qin = q
        q = qin + 1.0


def add_rank(out: IntField):
    from __externals__ import rank

    with computation(PARALLEL):
        with interval(rank, rank + 1):
            out = rank + 1.0


def setup_data_vars():
    shape = (7, 7, 3)
    q = make_storage_from_shape_uncached(shape)
    q[:] = 1.0
    q_ref = make_storage_from_shape_uncached(shape)
    q_ref[:] = 1.0
    return q, q_ref


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
@pytest.mark.parametrize("backend", ("numpy", "gtx86"))
@pytest.mark.parametrize("rebuild", (True, False))
@pytest.mark.parametrize("use_wrapper", (True, False))
def test_future_stencil(backend: str, rebuild: bool, use_wrapper: bool):
    class StencilWrapper:
        def __init__(self):
            self.stencil_object = None

        def __call__(self, *args, **kwargs) -> None:
            self.stencil_object(*args, **kwargs)

    set_backend(backend)
    StencilTable.clear()

    origin = (1, 1, 0)
    domain = (2, 2, 3)

    wrapper = StencilWrapper() if use_wrapper else None

    add1_object = future_stencil(
        definition=add1_stencil,
        backend=backend,
        rebuild=rebuild,
        wrapper=wrapper,
    )
    assert isinstance(add1_object, FutureStencil)

    # Fetch `field_info` to force a build
    field_info = add1_object.field_info
    assert len(field_info) == 1

    if use_wrapper:
        add1_object = wrapper.stencil_object
        assert isinstance(add1_object, StencilObject)

    q, q_ref = setup_data_vars()
    add1_object(q, origin=origin, domain=domain)
    q_ref[1:3, 1:3, :] = 2.0
    assert np.array_equal(q, q_ref)


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_distributed_table():
    comm = MPI.COMM_WORLD
    node_id = comm.Get_rank()
    n_nodes = comm.Get_size()

    table = StencilTable(comm, n_nodes)

    rand.seed(node_id)
    random_int = rand.randint(0, n_nodes)
    table[node_id] = random_int

    time.sleep(0.1)

    # Read from all other ranks
    expected_values = []
    received_values = []
    for n in range(n_nodes):
        rand.seed(n)
        expected_values.append(rand.randint(0, n_nodes))
        received_values.append(table[n])

    assert received_values == expected_values


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_one_sided_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data_type = MPI.FLOAT
    np_dtype = np.int32
    item_size = data_type.Get_size()

    n_ranks = comm.Get_size() + 1
    win_size = n_ranks * item_size if rank == 0 else 0
    win = MPI.Win.Allocate(
        size=win_size,
        disp_unit=item_size,
        comm=comm,
    )
    if rank == 0:
        mem = np.frombuffer(win, dtype=np_dtype)
        mem[:] = np.arange(len(mem), dtype=np_dtype)
    comm.Barrier()

    buffer = np.zeros(3, dtype=np_dtype)
    target = (rank, 2, data_type)
    win.Lock(rank=0)
    win.Get(buffer, target_rank=0, target=target)
    win.Unlock(rank=0)
    assert np.all(buffer == [rank, rank + 1, 0])


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_rank_adder_numpy():
    run_rank_adder_test("numpy", True)


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_rank_adder_gridtools():
    run_rank_adder_test("gtx86", True)


def run_rank_adder_test(backend: str, rebuild: bool):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    set_backend(backend)
    StencilTable.clear()

    origin = (0, 0, 0)
    domain = (1, 1, size)

    out_field = gt_storage.zeros(
        shape=domain, default_origin=origin, dtype=np.int64, backend=backend
    )
    ref_field = np.arange(1, size + 1, dtype=np.int64)

    for rank in range(0, size):
        stencil_object = future_stencil(
            definition=add_rank,
            backend=backend,
            rebuild=rebuild,
            externals={"rank": rank},
        )
        stencil_object(out_field, domain=domain, origin=origin)

    assert np.array_equal(out_field[0, 0, :], ref_field)


def create_future_stencil(definition: Callable = add_rank, backend="numpy") -> FutureStencil:
    origin: Tuple[int, int, int] = (0, 0, 0)
    domain: Tuple[int, int, int] = (1, 1, 10)
    out_field = gt_storage.zeros(
        shape=domain, default_origin=origin, dtype=np.int64, backend=backend
    )

    stencil_object = future_stencil(
        definition=definition,
        backend=backend,
        rebuild=True,
        externals={"rank": 0},
    )
    stencil_object(out_field, domain=domain, origin=origin)

    return stencil_object


def get_temp_dir() -> str:
    return "%s/.gt_cache" % ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
def test_stencil_serialization():
    future_stencil = create_future_stencil(backend="numpy")

    # Serialize stencil to numpy byte array
    bytes_array = future_stencil.serialize()
    assert isinstance(bytes_array, np.ndarray)
    assert bytes_array.size == 1655

    # Deserialize the bytes array into a stencil object
    deserialized_stencil = future_stencil.deserialize(bytes_array)
    stencil_object = future_stencil.stencil_object
    assert stencil_object._file_name == deserialized_stencil._file_name

    # TODO(eddied): Next steps...
    #   1. How to transmit the serialized numpy bytes array via MPI one sided-comm
    #      * Add send/recv abstract methods to StencilTable?
    #   2. Redirect cache directory to /dev/shm for writing stencils
    #   3. One node writes to scratch filesystem (original .gt_cache dir), stencil_id % n_nodes?


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
def test_sequential_transmission():
    # Redirect cache to temporary directory
    gt_cache_dir_name: str = gt.config.cache_settings["dir_name"]
    gt.config.cache_settings["dir_name"] = get_temp_dir()

    # Serialize stencil to numpy byte array
    future_stencil = create_future_stencil(backend="numpy")
    stencil_bytes = future_stencil.serialize()

    # Upload stencil bytes to stencil table...
    stencil_table = future_stencil._id_table
    stencil_table.write_stencil(stencil_bytes)

    # Remove local cache directory to simulate other node...
    stencil_object = future_stencil.stencil_object
    stencil_path = os.path.dirname(stencil_object._file_name)
    shutil.rmtree(stencil_path)

    # Fetch stencil from stencil table
    received_bytes = stencil_table.read_stencil()
    np.testing.assert_array_equal(stencil_bytes, received_bytes)

    # Deserialize stencil from received bytes
    deserialized_stencil = future_stencil.deserialize(received_bytes)
    assert stencil_object._file_name == deserialized_stencil._file_name

    # Restore cache location
    gt.config.cache_settings["dir_name"] = gt_cache_dir_name


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
def test_parallel_transmission():
    pass
