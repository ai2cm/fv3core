import unittest.mock

import gt4py.gtscript
import numpy as np
import pytest
from gt4py.gtscript import PARALLEL, computation, interval, stencil

from fv3core.decorators import FrozenStencil, StencilConfig
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_copy_wrapper(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_in[:] = 1.0
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    q_out[:] = 2.0
    stencil(q_in, q_out)
    np.testing.assert_array_equal(q_in, q_out)


@pytest.mark.parametrize(
    "rebuild, validate_args, format_source, device_sync",
    [[False, False, False, False], [True, False, False, False]],
)
def test_stencil_kwargs_passed_to_init(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    try:
        gt4py.gtscript.stencil = mock_stencil
        FrozenStencil(
            copy_stencil,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
            stencil_config=config,
            externals={},
        )
        mock_stencil.assert_called_once_with(
            definition=copy_stencil, externals={}, **config.stencil_kwargs
        )
    finally:
        gt4py.gtscript.stencil = stencil


def one_field_stencil(q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = 1.0


def three_field_stencil(q_out: FloatField, q_in1: FloatField, q_in2: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in1 + q_in2


def three_field_parameter_stencil(
    q_out: FloatField, q_in1: FloatField, q_in2: FloatField, param: float
):
    with computation(PARALLEL), interval(...):
        q_out = param * (q_in1 + q_in2)


@pytest.mark.parametrize(
    "definition, field_names",
    [
        [one_field_stencil, ("q_out",)],
        [three_field_stencil, ("q_out", "q_in1", "q_in2")],
        [three_field_parameter_stencil, ("q_out", "q_in1", "q_in2")],
    ],
)
def test_field_names(definition, field_names, backend):
    config = StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    result = FrozenStencil(
        definition,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    assert result.field_names == field_names


@pytest.mark.parametrize(
    "definition, parameter_names",
    [
        [one_field_stencil, tuple()],
        [three_field_stencil, tuple()],
        [three_field_parameter_stencil, ("param",)],
    ],
)
def test_parameter_names(definition, parameter_names, backend):
    config = StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    result = FrozenStencil(
        definition,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    assert result.parameter_names == parameter_names


def field_after_parameter_stencil(q_in: FloatField, param: float, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = param * q_in


def test_field_after_parameter_raises(backend):
    config = StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    with pytest.raises(TypeError):
        FrozenStencil(
            field_after_parameter_stencil,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
            stencil_config=config,
            externals={},
        )
