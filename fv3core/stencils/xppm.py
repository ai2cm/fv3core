from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import floor_cap, sign
from fv3core.utils.typing import FloatField

from .yppm import (
    c1,
    c2,
    c3,
    fx1_c_negative,
    get_b0,
    get_bl,
    is_smt5_mord5,
    is_smt5_most_mords,
    p1,
    p2,
    pert_ppm,
    pert_ppm_standard_constraint_fcn,
    s11,
    s14,
    s15,
)


sd = utils.sd
origin = (2, 0, 0)


@gtstencil(externals={"p1": p1, "p2": p2})
def main_al(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[-1, 0, 0] + q) + p2 * (q[-2, 0, 0] + q[1, 0, 0])


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_0(q: FloatField, dxa: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_1(q: FloatField, dxa: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = 0.5 * (
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


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_2(q: FloatField, dxa: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c3 * q[-1, 0, 0] + c2 * q[0, 0, 0] + c1 * q[1, 0, 0]


@gtscript.function
def get_br(al, q):
    br = al[1, 0, 0] - q
    return br


@gtscript.function
def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[-1, 0, 0] - c * b0[-1, 0, 0])


@gtscript.function
def flux_intermediates(q, al, mord):
    bl = get_bl(al=al, q=q)
    br = get_br(al=al, q=q)
    b0 = get_b0(bl=bl, br=br)
    smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
    tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
    return bl, br, b0, tmp


@gtscript.function
def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)


@gtscript.function
def final_flux(c, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtstencil()
def get_flux(
    q: FloatField, c: FloatField, al: FloatField, flux: FloatField, *, mord: int
):
    with computation(PARALLEL), interval(0, None):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        fx1 = fx1_fn(c, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug gets fixed
        flux = final_flux(c, q, fx1, tmp)  # noqa
        # bl = get_bl(al=al, q=q)
        # br = get_br(al=al, q=q)
        # b0 = get_b0(bl=bl, br=br)
        # smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        # tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
        # fx1 = fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)
        # flux = q[-1, 0, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtstencil()
def finalflux_ord8plus(
    q: FloatField, c: FloatField, bl: FloatField, br: FloatField, flux: FloatField
):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = q[-1, 0, 0] + fx1 if c > 0.0 else q + fx1


def dm_iord8plus_fcn(q: FloatField):
    xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
    dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
    dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@gtstencil()
def dm_iord8plus(q: FloatField, al: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        dm = dm_iord8plus_fcn(q)


def al_iord8plus_fcn(q: FloatField, dm: FloatField):
    return 0.5 * (q[-1, 0, 0] + q) + 1.0 / 3.0 * (dm[-1, 0, 0] - dm)


@gtstencil()
def al_iord8plus(q: FloatField, al: FloatField, dm: FloatField, r3: float):
    with computation(PARALLEL), interval(...):
        al = al_iord8plus_fcn(q, dm)


def blbr_iord8_fcn(q: FloatField, al: FloatField, dm: FloatField):
    # al, dm = al_iord8plus_fn(q, al, dm, r3)
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


@gtstencil()
def blbr_iord8(
    q: FloatField, al: FloatField, bl: FloatField, br: FloatField, dm: FloatField
):
    with computation(PARALLEL), interval(...):
        bl, br = blbr_iord8_fcn(q, al, dm)


@gtscript.function
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)
        + ((2.0 * dxa[1, 0, 0] + dxa[2, 0, 0]) * q[1, 0, 0] - dxa[1, 0, 0] * q[2, 0, 0])
        / (dxa[1, 0, 0] + dxa[2, 0, 0])
    )


@gtscript.function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        (
            (2.0 * dxa[-1, 0, 0] + dxa[-2, 0, 0]) * q[-1, 0, 0]
            - dxa[-1, 0, 0] * q[-2, 0, 0]
        )
        / (dxa[-2, 0, 0] + dxa[-1, 0, 0])
        + ((2.0 * dxa + dxa[1, 0, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0, 0])
    )


@gtscript.function
def xt_dxa_edge_0(q, dxa, xt_minmax):
    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa, xt_minmax):
    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtstencil()
def west_edge_iord8plus_0(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    bl: FloatField,
    br: FloatField,
    xt_minmax: bool,
):
    with computation(PARALLEL), interval(...):
        bl = s14 * dm[-1, 0, 0] + s11 * (q[-1, 0, 0] - q)
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q


@gtstencil()
def west_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    bl: FloatField,
    br: FloatField,
    xt_minmax: bool,
):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        xt = s15 * q + s11 * q[1, 0, 0] - s14 * dm[1, 0, 0]
        br = xt - q


@gtstencil()
def west_edge_iord8plus_2(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    al: FloatField,
    bl: FloatField,
    br: FloatField,
):
    with computation(PARALLEL), interval(...):
        xt = s15 * q[-1, 0, 0] + s11 * q - s14 * dm
        bl = xt - q
        br = al[1, 0, 0] - q


@gtstencil()
def east_edge_iord8plus_0(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    al: FloatField,
    bl: FloatField,
    br: FloatField,
):
    with computation(PARALLEL), interval(...):
        bl = al - q
        xt = s15 * q[1, 0, 0] + s11 * q + s14 * dm
        br = xt - q


@gtstencil()
def east_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    bl: FloatField,
    br: FloatField,
    xt_minmax: bool,
):
    with computation(PARALLEL), interval(...):
        xt = s15 * q + s11 * q[-1, 0, 0] + s14 * dm[-1, 0, 0]
        bl = xt - q
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q


@gtstencil()
def east_edge_iord8plus_2(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
    bl: FloatField,
    br: FloatField,
    xt_minmax: bool,
):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        br = s11 * (q[1, 0, 0] - q) - s14 * dm[1, 0, 0]


def compute_al(q, dxa, iord, is1, ie3, jfirst, jlast, kstart=0, nk=None):
    if nk is None:
        nk = spec.grid.npz - kstart
    dimensions = q.shape
    local_origin = (origin[0], origin[1], kstart)
    al = utils.make_storage_from_shape(dimensions, local_origin)
    domain_y = (1, dimensions[1], nk)
    if iord < 8:
        main_al(
            q,
            al,
            origin=(is1, jfirst, kstart),
            domain=(ie3 - is1 + 1, jlast - jfirst + 1, nk),
        )
        if not spec.grid.nested and spec.namelist.grid_type < 3:
            if spec.grid.west_edge:
                al_y_edge_0(
                    q, dxa, al, origin=(spec.grid.is_ - 1, 0, kstart), domain=domain_y
                )
                al_y_edge_1(
                    q, dxa, al, origin=(spec.grid.is_, 0, kstart), domain=domain_y
                )
                al_y_edge_2(
                    q, dxa, al, origin=(spec.grid.is_ + 1, 0, kstart), domain=domain_y
                )
            if spec.grid.east_edge:
                al_y_edge_0(
                    q, dxa, al, origin=(spec.grid.ie, 0, kstart), domain=domain_y
                )
                al_y_edge_1(
                    q, dxa, al, origin=(spec.grid.ie + 1, 0, kstart), domain=domain_y
                )
                al_y_edge_2(
                    q, dxa, al, origin=(spec.grid.ie + 2, 0, kstart), domain=domain_y
                )
        if iord < 0:
            floor_cap(
                al,
                0.0,
                origin=(spec.grid.is_ - 1, jfirst, kstart),
                domain=(spec.grid.nic + 3, jlast - jfirst + 1, nk),
            )
    return al


def compute_blbr_ord8plus(q, iord, jfirst, jlast, is1, ie1, kstart, nk):
    r3 = 1.0 / 3.0
    grid = spec.grid
    local_origin = (origin[0], origin[1], kstart)
    bl = utils.make_storage_from_shape(q.shape, local_origin)
    br = utils.make_storage_from_shape(q.shape, local_origin)
    dm = utils.make_storage_from_shape(q.shape, local_origin)
    al = utils.make_storage_from_shape(q.shape, local_origin)
    dj = jlast - jfirst + 1
    dm_iord8plus(
        q, al, dm, origin=(grid.is_ - 2, jfirst, kstart), domain=(grid.nic + 4, dj, nk)
    )
    al_iord8plus(
        q, al, dm, r3, origin=(is1, jfirst, kstart), domain=(ie1 - is1 + 2, dj, nk)
    )
    if iord == 8:
        blbr_iord8(
            q,
            al,
            bl,
            br,
            dm,
            origin=(is1, jfirst, kstart),
            domain=(ie1 - is1 + 1, dj, nk),
        )
    else:
        raise Exception("Unimplemented iord=" + str(iord))

    if spec.namelist.grid_type < 3 and not (grid.nested or spec.namelist.regional):
        y_edge_domain = (1, dj, nk)
        do_xt_minmax = True
        if grid.west_edge:
            west_edge_iord8plus_0(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.is_ - 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            west_edge_iord8plus_1(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.is_, jfirst, kstart),
                domain=y_edge_domain,
            )
            west_edge_iord8plus_2(
                q,
                grid.dxa,
                dm,
                al,
                bl,
                br,
                origin=(grid.is_ + 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            pert_ppm(q, bl, br, 1, grid.is_ - 1, jfirst, kstart, 3, dj, nk)
        if grid.east_edge:
            east_edge_iord8plus_0(
                q,
                grid.dxa,
                dm,
                al,
                bl,
                br,
                origin=(grid.ie - 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            east_edge_iord8plus_1(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.ie, jfirst, kstart),
                domain=y_edge_domain,
            )
            east_edge_iord8plus_2(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.ie + 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            pert_ppm(q, bl, br, 1, grid.ie - 1, jfirst, kstart, 3, dj, nk)
        return bl, br


def compute_al_fcn(q: FloatField, dxa: FloatField):
    from __externals__ import c1, c2, c3, i_end, i_start, iord, p1, p2

    assert __INLINED(iord < 8), "The code in this function requires iord < 8"

    al = p1 * (q[-1, 0, 0] + q) + p2 * (q[-2, 0, 0] + q[1, 0, 0])

    if __INLINED(iord < 0):
        assert __INLINED(False), "Not tested"
        al = max(al, 0.0)

    with horizontal(region[i_start - 1, :]):
        al = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q
    with horizontal(region[i_start, :]):
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
    with horizontal(region[i_start + 1, :]):
        al = c3 * q[-1, 0, 0] + c2 * q[0, 0, 0] + c1 * q[1, 0, 0]

    with horizontal(region[i_end, :]):
        al = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q
    with horizontal(region[i_end + 1, :]):
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
    with horizontal(region[i_end + 2, :]):
        al = c3 * q[-1, 0, 0] + c2 * q[0, 0, 0] + c1 * q[1, 0, 0]

    return al


def get_flux_fcn(q: FloatField, c: FloatField, al: FloatField):
    from __externals__ import mord

    bl = get_bl(al, q)
    br = get_br(al, q)
    b0 = get_b0(bl, br)

    if __INLINED(mord == 5):
        smt5 = is_smt5_mord5(bl, br)
    else:
        smt5 = is_smt5_most_mords(bl, br, b0)

    if smt5[-1, 0, 0]:
        tmp = smt5[-1, 0, 0]
    else:
        tmp = smt5[-1, 0, 0] + smt5

    fx1 = fx1_fn(c, br, b0, bl)
    return final_flux(c, q, fx1, tmp)  # noqa


def compute_blbr_ord8plus_fcn(q: FloatField, dxa: FloatField):
    from __externals__ import do_xt_minmax, i_end, i_start, iord, namelist

    dm = dm_iord8plus_fcn(q)
    al = al_iord8plus_fcn(q, dm)

    assert __INLINED(iord == 8), "Not yet implemented"
    # {
    bl, br = blbr_iord8_fcn(q, al, dm)
    # }

    # TODO: Use do_xt_minmax when gt4py BuiltinLiteral bug is fixed.
    assert __INLINED(namelist.grid_type < 3), "Remainder of function assumes this."
    # {
    with horizontal(region[i_start - 1, :]):
        bl = s14 * dm[-1, 0, 0] + s11 * (q[-1, 0, 0] - q)
        xt = xt_dxa_edge_0(q, dxa, do_xt_minmax)
        br = xt - q

    with horizontal(region[i_start, :]):
        xt = xt_dxa_edge_1(q, dxa, do_xt_minmax)
        bl = xt - q
        xt = s15 * q + s11 * q[1, 0, 0] - s14 * dm[1, 0, 0]
        br = xt - q

    with horizontal(region[i_start + 1, :]):
        xt = s15 * q[-1, 0, 0] + s11 * q - s14 * dm
        bl = xt - q
        br = al[1, 0, 0] - q

    with horizontal(region[i_end - 1, :]):
        bl = al - q
        xt = s15 * q[1, 0, 0] + s11 * q + s14 * dm
        br = xt - q

    with horizontal(region[i_end, :]):
        xt = s15 * q + s11 * q[-1, 0, 0] + s14 * dm[-1, 0, 0]
        bl = xt - q
        xt = xt_dxa_edge_0(q, dxa, do_xt_minmax)
        br = xt - q

    with horizontal(region[i_end + 1, :]):
        xt = xt_dxa_edge_1(q, dxa, do_xt_minmax)
        bl = xt - q
        br = s11 * (q[1, 0, 0] - q) - s14 * dm[1, 0, 0]
    # }

    bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    return bl, br


def get_flux_ord8plus_fcn(q: FloatField, c: FloatField, bl: FloatField, br: FloatField):
    b0 = get_b0(bl, br)
    fx1 = fx1_fn(c, br, b0, bl)

    return q[-1, 0, 0] + fx1 if c > 0.0 else q[0, 0, 0] + fx1


def _compute_flux_stencil(
    q: FloatField, c: FloatField, dxa: FloatField, xflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al_fcn(q, dxa)
            xflux = get_flux_fcn(q, c, al)
        else:
            bl, br = compute_blbr_ord8plus_fcn(q, dxa)
            xflux = get_flux_ord8plus_fcn(q, c, bl, br)


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
            "p1": p1,
            "p2": p2,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "do_xt_minmax": True,
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
