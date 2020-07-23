import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
from gt4py.gtscript import computation, interval, PARALLEL


@utils.stencil()
def copy_stencil(q_in: utils.sd, q_out: utils.sd):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def copy(q_in, origin, domain=None):
    q_out = utils.make_storage_from_shape(q_in.shape, origin)
    copy_stencil(q_in, q_out, origin=origin, domain=domain)
    return q_out
