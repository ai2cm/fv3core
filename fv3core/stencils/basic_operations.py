import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.utils.gt4py_utils as utils
from fv3core.utils.types import Field, Int3


# The stencils here are compiled for this data type
DATA_TYPE = float
DataField = Field[DATA_TYPE]


@utils.stencil()
def copy_stencil(q_in: DataField, q_out: DataField):
    """Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def copy(q_in: DataField, *, origin: Int3 = (0, 0, 0), domain: Int3 = None):
    """Copy q_in to a field returned.

    Args:
        q_in: input field
        origin: Origin of the copy and new field
        domain: Extent to copy

    Returns:
        types.Field[float]: Copied field
    """
    q_out = utils.make_storage_from_shape(q_in.shape, origin)
    copy_stencil(q_in, q_out, origin=origin, domain=domain)
    return q_out


@utils.stencil()
def adjustmentfactor_stencil(adjustment: DataField, q_out: DataField):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out * adjustment


@utils.stencil()
def adjust_divide_stencil(adjustment: DataField, q_out: DataField):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out / adjustment


@utils.stencil()
def multiply_stencil(in1: DataField, in2: DataField, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 * in2


@utils.stencil()
def divide_stencil(in1: DataField, in2: DataField, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 / in2


@utils.stencil()
def addition_stencil(in1: DataField, in2: DataField, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 + in2


@utils.stencil()
def add_term_stencil(in1: DataField, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = out + in1


@utils.stencil()
def add_term_two_vars(in1: DataField, out1: DataField, in2: DataField, out2: DataField):
    with computation(PARALLEL), interval(...):
        out1[0, 0, 0] = out1 + in1
        out2[0, 0, 0] = out2 + in2


@utils.stencil()
def subtract_term_stencil(in1: DataField, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = out - in1


@utils.stencil()
def multiply_constant(in1: DataField, in2: float, out: DataField):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 * in2


@utils.stencil()
def multiply_constant_inout(inout: DataField, in_float: float):
    with computation(PARALLEL), interval(...):
        inout[0, 0, 0] = in_float * inout


@utils.stencil()
def floor_cap(var: DataField, floor_value: float):
    with computation(PARALLEL), interval(0, None):
        var[0, 0, 0] = var if var > floor_value else floor_value


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value


@gtscript.function
def sign(a, b):
    asignb = absolute_value(a)
    if b > 0:
        asignb = asignb
    else:
        asignb = -asignb
    return asignb


@gtscript.function
def min_fn(a, b):
    return a if a < b else b


@gtscript.function
def max_fn(a, b):
    return a if a > b else b


@gtscript.function
def dim(a, b):
    diff = a - b if a - b > 0 else 0
    return diff
