from typing import Optional

from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils import yppm
from fv3core.stencils.basic_operations import sign
from fv3core.utils.typing import FloatField


def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[-1, 0, 0] - c * b0[-1, 0, 0])


def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else yppm.fx1_c_negative(c, bl, b0)


def final_flux(c, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


def dm_iord8plus(q: FloatField):
    xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
    dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
    dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


def al_iord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[-1, 0, 0] + q) + 1.0 / 3.0 * (dm[-1, 0, 0] - dm)


def blbr_iord8(q: FloatField, al: FloatField, dm: FloatField):
    # al, dm = al_iord8plus_fn(q, al, dm, r3)
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)
        + ((2.0 * dxa[1, 0, 0] + dxa[2, 0, 0]) * q[1, 0, 0] - dxa[1, 0, 0] * q[2, 0, 0])
        / (dxa[1, 0, 0] + dxa[2, 0, 0])
    )


def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        (
            (2.0 * dxa[-1, 0, 0] + dxa[-2, 0, 0]) * q[-1, 0, 0]
            - dxa[-1, 0, 0] * q[-2, 0, 0]
        )
        / (dxa[-2, 0, 0] + dxa[-1, 0, 0])
        + ((2.0 * dxa + dxa[1, 0, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0, 0])
    )


def xt_dxa_edge_0(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


def xt_dxa_edge_1(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


def west_edge_iord8plus_0(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    bl = yppm.s14 * dm[-1, 0, 0] + yppm.s11 * (q[-1, 0, 0] - q)
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


def west_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    xt = yppm.s15 * q + yppm.s11 * q[1, 0, 0] - yppm.s14 * dm[1, 0, 0]
    br = xt - q
    return bl, br


def west_edge_iord8plus_2(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    xt = yppm.s15 * q[-1, 0, 0] + yppm.s11 * q - yppm.s14 * dm
    bl = xt - q
    br = al[1, 0, 0] - q
    return bl, br


def east_edge_iord8plus_0(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    bl = al - q
    xt = yppm.s15 * q[1, 0, 0] + yppm.s11 * q + yppm.s14 * dm
    br = xt - q
    return bl, br


def east_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = yppm.s15 * q + yppm.s11 * q[-1, 0, 0] + yppm.s14 * dm[-1, 0, 0]
    bl = xt - q
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


def east_edge_iord8plus_2(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    br = yppm.s11 * (q[1, 0, 0] - q) - yppm.s14 * dm[1, 0, 0]
    return bl, br


def compute_al(q: FloatField, dxa: FloatField):
    from __externals__ import i_end, i_start, iord

    assert __INLINED(iord < 8), "The code in this function requires iord < 8"

    al = yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])

    if __INLINED(iord < 0):
        assert __INLINED(False), "Not tested"
        al = max(al, 0.0)

    with horizontal(region[i_start - 1, :], region[i_end, :]):
        al = yppm.c1 * q[-2, 0, 0] + yppm.c2 * q[-1, 0, 0] + yppm.c3 * q
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        al = 0.5 * (
            (
                (2.0 * dxa[-1, 0, 0] + dxa[-2, 0, 0]) * q[-1, 0, 0]
                - dxa[-1, 0, 0] * q[-2, 0, 0]
            )
            / (dxa[-2, 0, 0] + dxa[-1, 0, 0])
            + (
                (2.0 * dxa[0, 0, 0] + dxa[1, 0, 0]) * q[0, 0, 0]
                - dxa[0, 0, 0] * q[1, 0, 0]
            )
            / (dxa[0, 0, 0] + dxa[1, 0, 0])
        )
    with horizontal(region[i_start + 1, :], region[i_end + 2, :]):
        al = yppm.c3 * q[-1, 0, 0] + yppm.c2 * q[0, 0, 0] + yppm.c1 * q[1, 0, 0]

    return al


def get_flux(q: FloatField, c: FloatField, al: FloatField):
    from __externals__ import mord

    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[1, 0, 0] - q[0, 0, 0]
    b0 = bl + br

    if __INLINED(mord == 5):
        smt5 = yppm.is_smt5_mord5(bl, br)
    else:
        smt5 = yppm.is_smt5_most_mords(bl, br, b0)

    if smt5[-1, 0, 0]:
        tmp = smt5[-1, 0, 0]
    else:
        tmp = smt5[-1, 0, 0] + smt5

    fx1 = fx1_fn(c, br, b0, bl)
    return final_flux(c, q, fx1, tmp)  # noqa


def compute_blbr_ord8plus(q: FloatField, dxa: FloatField):
    from __externals__ import i_end, i_start, iord, namelist

    dm = dm_iord8plus(q)
    al = al_iord8plus(q, dm)

    assert __INLINED(iord == 8), "Unimplemented iord"
    # {
    bl, br = blbr_iord8(q, al, dm)
    # }

    assert __INLINED(namelist.grid_type < 3)
    # {
    with horizontal(region[i_start - 1, :]):
        bl, br = west_edge_iord8plus_0(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_start, :]):
        bl, br = west_edge_iord8plus_1(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_start + 1, :]):
        bl, br = west_edge_iord8plus_2(q, dm, al)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end - 1, :]):
        bl, br = east_edge_iord8plus_0(q, dm, al)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end, :]):
        bl, br = east_edge_iord8plus_1(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end + 1, :]):
        bl, br = east_edge_iord8plus_2(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)
    # }

    return bl, br


def get_flux_ord8plus(q: FloatField, c: FloatField, bl: FloatField, br: FloatField):
    b0 = bl + br
    fx1 = fx1_fn(c, br, b0, bl)

    return q[-1, 0, 0] + fx1 if c > 0.0 else q[0, 0, 0] + fx1


def _compute_flux_stencil(
    q: FloatField, c: FloatField, dxa: FloatField, xflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dxa)
            xflux = get_flux(q, c, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dxa)
            xflux = get_flux_ord8plus(q, c, bl, br)


def compute_flux(
    q: FloatField,
    c: FloatField,
    xflux: FloatField,
    iord: int,
    jfirst: int,
    jlast: int,
    kstart: int = 0,
    nk: Optional[int] = None,
):
    """
    Compute x-flux using the PPM method.

    Args:
        q (in): Transported scalar
        c (in): Courant number
        xflux (out): Flux
        iord: Method selector
        jfirst: Starting index of the J-dir compute domain
        jlast: Final index of the J-dir compute domain
        kstart: First index of the K-dir compute domain
        nk: Number of indices in the K-dir compute domain
    """
    # Tests: xppm, fvtp2d, tracer2d1l
    grid = spec.grid
    if nk is None:
        nk = spec.grid.npz - kstart
    stencil = gtstencil(
        definition=_compute_flux_stencil,
        externals={
            "iord": iord,
            "mord": abs(iord),
            "xt_minmax": True,
        },
    )
    nj = jlast - jfirst + 1
    stencil(
        q,
        c,
        grid.dxa,
        xflux,
        origin=(grid.is_, jfirst, kstart),
        domain=(grid.nic + 1, nj, nk),
    )
