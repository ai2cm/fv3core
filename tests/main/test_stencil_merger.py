from typing import Tuple

import numpy as np
import pytest
from gt4py.gtscript import PARALLEL, computation, interval, stencil

from fv3core.decorators import disable_merge_stencils, enable_merge_stencils
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.mpi import MPI
from fv3core.utils.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add1_stencil(q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out += 1.0


def setup_data_vars(num_storages: int = 3, init_val: float = 1.0):
    shape = (7, 7, 3)
    storages = []
    for n in range(num_storages):
        storage = make_storage_from_shape_uncached(shape)
        storage[:] = init_val
        storages.append(storage)

    return tuple(storages)


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
@pytest.mark.parametrize("backend", ("numpy", "gtx86"))
@pytest.mark.parametrize("rebuild", (True, False))
@pytest.mark.parametrize("do_merge", (False, True))
def test_stencil_merger(backend: str, rebuild: bool, do_merge: bool):
    set_backend(backend)

    q_in, q_out, q_ref = setup_data_vars()
    q_ref[1:3, 1:3, :] = 2.0

    origin = (1, 1, 0)
    domain = (2, 2, 3)

    q_out = run_stencil_test(q_in, q_out, origin, domain, backend, rebuild, do_merge)
    assert np.array_equal(q_out, q_ref)


def run_stencil_test(
    q_in: FloatField,
    q_out: FloatField,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
    backend: str,
    rebuild: bool,
    do_merge: bool,
):
    if do_merge:
        enable_merge_stencils()

    copy_object = stencil(
        definition=copy_stencil,
        backend=backend,
        rebuild=rebuild,
    )

    add1_object = stencil(
        definition=add1_stencil,
        backend=backend,
        rebuild=rebuild,
    )

    if do_merge:
        disable_merge_stencils()

    copy_object(q_in, q_out, origin=origin, domain=domain)
    add1_object(q_out, origin=origin, domain=domain)

    return q_out
