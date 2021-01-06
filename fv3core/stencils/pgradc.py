import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


FloatField = utils.FloatField


@gtscript.function
def p_grad_c_u(
    uc_in: FloatField,
    wk: FloatField,
    pkc: FloatField,
    gz: FloatField,
    rdxc: FloatField,
    dt2: float,
):
    return uc_in + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
        (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
        + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
    )


@gtscript.function
def get_wk(pkc: FloatField, delpc: FloatField, hydrostatic: int):
    return pkc[0, 0, 1] - pkc if hydrostatic else delpc


@gtscript.function
def p_grad_c_u_wk(
    uc_in: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    rdxc: FloatField,
    hydrostatic: int,
    dt2: float,
):
    wk = get_wk(pkc, delpc, hydrostatic)
    return p_grad_c_u(uc_in, wk, pkc, gz, rdxc, dt2)


@gtscript.function
def p_grad_c_v(vc_in, wk, pkc, gz, rdyc, dt2):
    return vc_in + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
        (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
        + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
    )


@gtscript.function
def p_grad_c_v_wk(vc_in, delpc, pkc, gz, rdyc, hydrostatic, dt2):
    wk = get_wk(pkc, delpc, hydrostatic)
    return p_grad_c_v(vc_in, wk, pkc, gz, rdyc, dt2)


@gtstencil()
def p_grad_c(
    uc_in: FloatField,
    vc_in: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    hydrostatic: int,
    dt2: float,
):
    from __externals__ import i_end, j_end

    with computation(PARALLEL), interval(0, -1):
        with parallel(region[:, : j_end + 1]):
            uc_in = p_grad_c_u_wk(
                uc_in, delpc, pkc, gz, rdxc, hydrostatic, dt2
            )  # TODO: add [0, 0, 0] when gt4py bug is fixed
        with parallel(region[: i_end + 1, :]):
            vc_in = p_grad_c_v_wk(
                vc_in, delpc, pkc, gz, rdyc, hydrostatic, dt2
            )  # TODO: add [0, 0, 0] when gt4py bug is fixed


def compute(
    uc: FloatField,
    vc: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    dt2: float,
):
    grid = spec.grid

    p_grad_c(
        uc,
        vc,
        delpc,
        pkc,
        gz,
        grid.rdxc,
        grid.rdyc,
        int(spec.namelist.hydrostatic),
        dt2,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz + 1),
    )
