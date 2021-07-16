import numpy as np
import pytest

from gt4py.gtscript import PARALLEL, computation, interval, stencil

from fv3core.decorators import StencilConfig
from fv3core.utils.future_stencil import FutureStencil
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
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
