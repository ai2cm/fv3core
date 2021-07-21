import datetime as dt
import numpy as np
import pytest
import random as rand
import time

from gt4py.gtscript import PARALLEL, computation, interval, stencil

from fv3core.decorators import StencilConfig
from fv3core.utils.future_stencil import FutureStencil, RedisTable, WindowTable
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.mpi import MPI
from fv3core.utils.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add1_stencil(q: FloatField):
    with computation(PARALLEL), interval(...):
        qin = q
        q = qin + 1.0


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
def test_future_stencil(backend: str, rebuild: bool):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    set_backend(backend)

    origin = (1, 1, 0)
    domain = (2, 2, 3)

    add1_object = stencil(
        definition=add1_stencil,
        defer_function=FutureStencil,
        **config.stencil_kwargs,
    )
    assert isinstance(add1_object, FutureStencil)
    assert not add1_object.is_built

    # Fetch `field_info` to force a build
    field_info = add1_object.field_info
    assert add1_object.is_built
    assert len(field_info) == 1

    q, q_ref = setup_data_vars()
    add1_object(q, origin=origin, domain=domain)
    q_ref[1:3, 1:3, :] = 2.0
    np.testing.assert_array_equal(q.data, q_ref.data)


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() == 1,
    reason="Not running in parallel with mpi",
)
# @pytest.mark.parametrize("table_type", ("redis", "window"))
@pytest.mark.parametrize("table_type", ("window",))
def test_distributed_table(table_type: str):
    comm = MPI.COMM_WORLD
    node_id = comm.Get_rank()
    n_nodes = comm.Get_size()

    table = RedisTable() if table_type == "redis" else WindowTable(comm, n_nodes)

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
    MPI is not None and MPI.COMM_WORLD.Get_size() == 1,
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
