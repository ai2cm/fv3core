import pytest
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.decorators
import fv3core.utils.gt4py_utils
import fv3core.utils.safety
from fv3core.utils.typing import FloatField


@pytest.fixture
def storage():
    return fv3core.utils.gt4py_utils.make_storage_from_shape_uncached((3, 3, 3))


@pytest.fixture(params=["frozen", "wrapped", "decorated"])
def stencil(request, storage):
    def func(storage: FloatField):
        with computation(PARALLEL), interval(...):
            storage = 0.0

    if request.param == "frozen":
        return fv3core.decorators.FrozenStencil(
            func, origin=(0, 0, 0), domain=storage.shape
        )
    elif request.param == "wrapped":
        return fv3core.decorators.StencilWrapper(func)
    elif request.param == "decorated":
        return fv3core.decorators.gtstencil()(func)
    else:
        raise NotImplementedError(request.param)


def test_access_storage_after_stencil_call_raises(storage, stencil):
    try:
        stencil(storage, origin=(0, 0, 0), domain=storage.shape)
    except TypeError:
        stencil(storage)
    with pytest.raises(fv3core.utils.safety.UnsafeAccess):
        storage[0, 0, 0]


def test_access_storage_after_sync(storage, stencil):
    try:
        stencil(storage, origin=(0, 0, 0), domain=storage.shape)
    except TypeError:
        stencil(storage)
    fv3core.utils.gt4py_utils.device_sync()
    storage[0, 0, 0]
