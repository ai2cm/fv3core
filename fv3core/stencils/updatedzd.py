import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy
from fv3core.stencils.fxadv import ra_x_func, ra_y_func
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


DZ_MIN = constants.DZ_MIN


@gtstencil()
def ra_x_stencil(area: FloatFieldIJ, xfx_adv: FloatField, ra_x: FloatField):
    with computation(PARALLEL), interval(...):
        ra_x = ra_x_func(area, xfx_adv)


@gtstencil()
def ra_y_stencil(area: FloatFieldIJ, yfx_adv: FloatField, ra_y: FloatField):
    with computation(PARALLEL), interval(...):
        ra_y = ra_y_func(area, yfx_adv)


@gtscript.function
def zh_base(z2, area, fx, fy, ra_x, ra_y):
    return (z2 * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (ra_x + ra_y - area)


@gtstencil()
def zh_damp_stencil(
    area: FloatFieldIJ,
    z2: FloatField,
    fx: FloatField,
    fy: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
    fx2: FloatField,
    fy2: FloatField,
    rarea: FloatField,
    zh: FloatField,
):
    with computation(PARALLEL), interval(...):
        zhbase = zh_base(z2, area, fx, fy, ra_x, ra_y)
        zh[0, 0, 0] = zhbase + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea


@gtstencil()
def zh_stencil(
    area: FloatField,
    zh: FloatField,
    fx: FloatField,
    fy: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
):
    with computation(PARALLEL), interval(...):
        zh = zh_base(zh, area, fx, fy, ra_x, ra_y)


# NOTE: We have not ported the uniform_grid True option as it is never called
# that way in this model. We have also ignored limite != 0 for the same reason.
@gtstencil()
def edge_profile(
    q1: FloatField,
    q2: FloatField,
    qe1: FloatField,
    qe2: FloatField,
    dp0: FloatFieldK,
    gam: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            g0 = dp0[1] / dp0
            xt1 = 2.0 * g0 * (g0 + 1.0)
            bet = g0 * (g0 + 0.5)
            qe1 = (xt1 * q1 + q1[0, 0, 1]) / bet
            qe2 = (xt1 * q2 + q2[0, 0, 1]) / bet
            gam = (1.0 + g0 * (g0 + 1.5)) / bet
        with interval(1, -1):
            gk = dp0[-1] / dp0
            bet = 2.0 + 2.0 * gk - gam
            qe1 = (3.0 * (q1[0, 0, -1] + gk * q1) - qe1[0, 0, -1]) / bet
            qe2 = (3.0 * (q2[0, 0, -1] + gk * q2) - qe2[0, 0, -1]) / bet
            gam = gk / bet
        with interval(-1, None):
            a_bot = 1.0 + gk[0, 0, -1] * (gk[0, 0, -1] + 1.5)
            xt1 = 2.0 * gk[0, 0, -1] * (gk[0, 0, -1] + 1.0)
            xt2 = gk[0, 0, -1] * (gk[0, 0, -1] + 0.5) - a_bot * gam
            qe1 = (xt1 * q1[0, 0, -1] + q1[0, 0, -2] - a_bot * qe1[0, 0, -1]) / xt2
            qe2 = (xt1 * q2[0, 0, -1] + q2[0, 0, -2] - a_bot * qe2[0, 0, -1]) / xt2
    with computation(BACKWARD), interval(0, -1):
        qe1 = qe1 - gam * qe1[0, 0, 1]
        qe2 = qe2 - gam * qe2[0, 0, 1]


def edge_python(q1, q2, qe1, qe2, dp0, gam, islice, jslice, qe1_2, gam_2):
    grid = spec.grid
    dcol = dp0[0, 0, :]

    km = grid.npz - 1
    g0 = dcol[1] / dcol[0]
    xt1 = 2.0 * g0 * (g0 + 1.0)
    bet = g0 * (g0 + 0.5)

    qe1[islice, jslice, 0] = (xt1 * q1[islice, jslice, 0] + q1[islice, jslice, 1]) / bet

    qe2[islice, jslice, 0] = (xt1 * q2[islice, jslice, 0] + q2[islice, jslice, 1]) / bet
    gam[islice, jslice, 0] = (1.0 + g0 * (g0 + 1.5)) / bet

    for k in range(1, km + 1):
        gk = dcol[k - 1] / dcol[k]
        bet = 2.0 + 2.0 * gk - gam[islice, jslice, k - 1]
        qe1[islice, jslice, k] = (
            3.0 * (q1[islice, jslice, k - 1] + gk * q1[islice, jslice, k])
            - qe1[islice, jslice, k - 1]
        ) / bet
        qe2[islice, jslice, k] = (
            3.0 * (q2[islice, jslice, k - 1] + gk * q2[islice, jslice, k])
            - qe2[islice, jslice, k - 1]
        ) / bet
        gam[islice, jslice, k] = gk / bet

    a_bot = 1.0 + gk * (gk + 1.5)
    xt1 = 2.0 * gk * (gk + 1.0)
    xt2 = gk * (gk + 0.5) - a_bot * gam[islice, jslice, km]
    qe1[islice, jslice, km + 1] = (
        xt1 * q1[islice, jslice, km]
        + q1[islice, jslice, km - 1]
        - a_bot * qe1[islice, jslice, km]
    ) / xt2
    qe2[islice, jslice, km + 1] = (
        xt1 * q2[islice, jslice, km]
        + q2[islice, jslice, km - 1]
        - a_bot * qe2[islice, jslice, km]
    ) / xt2
    for k in range(km, -1, -1):
        qe1[islice, jslice, k] = (
            qe1[islice, jslice, k] - gam[islice, jslice, k] * qe1[islice, jslice, k + 1]
        )
        qe2[islice, jslice, k] = (
            qe2[islice, jslice, k] - gam[islice, jslice, k] * qe2[islice, jslice, k + 1]
        )


@gtstencil()
def out(zs: FloatField, zh: FloatField, ws: FloatField, dt: float):
    with computation(BACKWARD):
        with interval(-1, None):
            ws[0, 0, 0] = (zs - zh) * 1.0 / dt
        with interval(0, -1):
            other = zh[0, 0, 1] + DZ_MIN
            zh[0, 0, 0] = zh if zh > other else other


def compute(
    ndif: FloatField,
    damp_vtd: FloatField,
    dp0: FloatField,
    zs: FloatField,
    zh: FloatField,
    crx: FloatField,
    cry: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    wsd: FloatField,
    dt: float,
):
    grid = spec.grid
    crx_adv = utils.make_storage_from_shape(
        crx.shape, grid.compute_origin(add=(0, -grid.halo, 0))
    )
    cry_adv = utils.make_storage_from_shape(
        cry.shape, grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    xfx_adv = utils.make_storage_from_shape(
        xfx.shape, grid.compute_origin(add=(0, -grid.halo, 0))
    )
    yfx_adv = utils.make_storage_from_shape(
        yfx.shape, grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    ra_x = utils.make_storage_from_shape(
        crx.shape, grid.compute_origin(add=(0, -grid.halo, 0))
    )
    ra_y = utils.make_storage_from_shape(
        cry.shape, grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    gam = utils.make_storage_from_shape(zs.shape, grid.full_origin())

    edge_profile(
        crx,
        xfx,
        crx_adv,
        xfx_adv,
        dp0,
        gam,
        origin=(grid.is_, grid.jsd, 0),
        domain=(grid.nic + 1, grid.njd, grid.npz + 1),
    )
    edge_profile(
        cry,
        yfx,
        cry_adv,
        yfx_adv,
        dp0,
        gam,
        origin=(grid.isd, grid.js, 0),
        domain=(grid.nid, grid.njc + 1, grid.npz + 1),
    )

    ra_x_stencil(
        grid.area,
        xfx_adv,
        ra_x,
        origin=grid.compute_origin(add=(0, -grid.halo, 0)),
        domain=(grid.nic, grid.njd, grid.npz + 1),
    )
    ra_y_stencil(
        grid.area,
        yfx_adv,
        ra_y,
        origin=grid.compute_origin(add=(-grid.halo, 0, 0)),
        domain=(grid.nid, grid.njc, grid.npz + 1),
    )

    ndif[-1] = ndif[-2]
    damp_vtd[-1] = damp_vtd[-2]
    col = {"ndif": ndif, "damp": damp_vtd}

    for ki, nk in utils.get_kstarts(col, grid.npz + 1):
        column_calls(
            zh,
            crx_adv,
            cry_adv,
            xfx_adv,
            yfx_adv,
            ra_x,
            ra_y,
            ndif[ki],
            damp_vtd[ki],
            ki,
            nk,
        )

    out(
        zs,
        zh,
        wsd,
        dt,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, grid.npz + 1),
    )


def column_calls(
    zh: FloatField,
    crx_adv: FloatField,
    cry_adv: FloatField,
    xfx_adv: FloatField,
    yfx_adv: FloatField,
    ra_x: FloatField,
    ra_y: FloatField,
    ndif: float,
    damp: float,
    kstart: int,
    nk: int,
):
    if damp <= 1e-5:
        raise Exception("untested")
    grid = spec.grid
    full_origin = (grid.isd, grid.jsd, kstart)
    compute_origin = (grid.is_, grid.js, kstart)
    compute_domain = (grid.nic, grid.njc, nk)

    wk = utils.make_storage_from_shape(zh.shape, full_origin)
    fx2 = utils.make_storage_from_shape(zh.shape, full_origin)
    fy2 = utils.make_storage_from_shape(zh.shape, full_origin)
    fx = utils.make_storage_from_shape(zh.shape, full_origin)
    fy = utils.make_storage_from_shape(zh.shape, full_origin)
    z2 = copy(zh, origin=full_origin, domain=(grid.nid, grid.njd, nk))

    fvtp2d.compute_no_sg(
        z2,
        crx_adv,
        cry_adv,
        spec.namelist.hord_tm,
        xfx_adv,
        yfx_adv,
        ra_x,
        ra_y,
        fx,
        fy,
        kstart=kstart,
        nk=nk,
    )
    delnflux.compute_no_sg(z2, fx2, fy2, int(ndif), damp, wk, kstart=kstart, nk=nk)
    zh_damp_stencil(
        grid.area,
        z2,
        fx,
        fy,
        ra_x,
        ra_y,
        fx2,
        fy2,
        grid.rarea,
        zh,
        origin=compute_origin,
        domain=compute_domain,
    )
