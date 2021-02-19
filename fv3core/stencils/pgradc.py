import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtstencil()
def p_grad_c_stencil(
    uc: FloatField,
    vc: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    dt2: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js, namelist

    with computation(PARALLEL), interval(...):
        if __INLINED(namelist.hydrostatic):
            wk = pkc[0, 0, 1] - pkc
        else:
            wk = delpc
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            uc = uc + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
                (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
                + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
            )
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            vc = vc + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
                (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
                + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
            )


def compute(uc, vc, delpc, pkc, gz, dt2):
    grid = spec.grid
    p_grad_c_stencil(
        uc,
        vc,
        delpc,
        pkc,
        gz,
        grid.rdxc,
        grid.rdyc,
        dt2=dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(1, 1, 0)),
    )
