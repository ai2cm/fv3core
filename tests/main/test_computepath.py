import contextlib
import unittest.mock

import gt4py.gtscript
import numpy as np
import pytest
from gt4py.gtscript import PARALLEL, computation, interval

from fv3core import StencilConfig
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.stencil import FrozenStencil, computepath_function
from fv3core.utils.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in + 1


@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("format_source", [False])
@pytest.mark.parametrize("device_sync", [True])
def test_computepath(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    if "dace" not in backend:
        pytest.skip(f"DaCe backend must be used, {backend} given")
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    print(backend)
    stencil1 = FrozenStencil(
        copy_stencil,
        origin=(3, 3, 0),
        domain=(12, 12, 79),
        stencil_config=config,
        externals={},
    )
    stencil2 = FrozenStencil(
        add_stencil,
        origin=(3, 3, 0),
        domain=(12, 12, 79),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    q_in[:] = 1.0
    q_interim = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    q_interim[:] = 2.0
    q_out = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    q_out[:] = 3.0

    import cupy as cp

    expected_interim = cp.asnumpy(q_interim.copy())
    expected_interim[3:15, 3:15, 0:79] = q_in[3:15, 3:15, 0:79]
    expected_out = cp.asnumpy(q_out.copy())
    expected_out[3:15, 3:15, 0:79] = expected_interim[3:15, 3:15, 0:79] + 1

    @computepath_function
    def orchestrated(q_in, q_interim, q_out):
        stencil1(q_in, q_interim)
        stencil2(q_interim, q_out)

    orchestrated(q_in, q_interim, q_out)

    assert np.allclose(q_interim, expected_interim)
    assert np.allclose(q_out, expected_out)


@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("format_source", [False])
@pytest.mark.parametrize("device_sync", [True])
def test_computepath_return(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    if "dace" not in backend:
        pytest.skip(f"DaCe backend must be used, {backend} given")
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    print(backend)
    stencil1 = FrozenStencil(
        copy_stencil,
        origin=(3, 3, 0),
        domain=(12, 12, 79),
        stencil_config=config,
        externals={},
    )
    stencil2 = FrozenStencil(
        add_stencil,
        origin=(3, 3, 0),
        domain=(12, 12, 79),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    q_in[:] = 1.0
    gq_interim = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    gq_interim[:] = 2.0
    q_out = make_storage_from_shape_uncached((19, 19, 80), (3, 3, 0))
    q_out[:] = 3.0

    import cupy as cp

    expected_interim = cp.asnumpy(gq_interim.copy())
    expected_interim[3:15, 3:15, 0:79] = q_in[3:15, 3:15, 0:79]
    expected_out = cp.asnumpy(q_out.copy())
    expected_out[3:15, 3:15, 0:79] = expected_interim[3:15, 3:15, 0:79] + 1

    def nested(q_in):
        stencil1(q_in, gq_interim)
        return gq_interim

    @computepath_function
    def orchestrated(q_in, q_out):
        q_interim = nested(q_in)
        stencil2(q_interim, q_out)

    orchestrated(q_in, q_out)

    assert np.allclose(gq_interim, expected_interim)
    assert np.allclose(q_out, expected_out)
