from typing import Optional

from gt4py import gtscript
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


input_vars = ["q", "c"]
inputs_params = ["jord", "ifirst", "ilast"]
output_vars = ["flux"]

# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

s11 = 11.0 / 14.0
s14 = 4.0 / 7.0
s15 = 3.0 / 14.0

sd = utils.sd
origin = (0, 2, 0)


def grid():
    return spec.grid


@gtstencil(externals={"p1": p1, "p2": p2})
def main_al_ord_under8(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_0(q: FloatField, dya: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_1(q: FloatField, dya: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = 0.5 * (
            (
                (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
                - dya[0, -1, 0] * q[0, -2, 0]
            )
            / (dya[0, -2, 0] + dya[0, -1, 0])
            + (
                (2.0 * dya[0, 0, 0] + dya[0, 1, 0]) * q[0, 0, 0]
                - dya[0, 0, 0] * q[0, 1, 0]
            )
            / (dya[0, 0, 0] + dya[0, 1, 0])
        )


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_x_under8_edge_2(q: FloatField, dya: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]


@gtscript.function
def get_bl(al, q):
    bl = al - q
    return bl


@gtscript.function
def get_br(al, q):
    br = al[0, 1, 0] - q
    return br


@gtscript.function
def get_b0(bl, br):
    b0 = bl + br
    return b0


@gtscript.function
def is_smt5_mord5(bl, br):
    return bl * br < 0


@gtscript.function
def is_smt5_most_mords(bl, br, b0):
    return (3.0 * abs(b0)) < abs(bl - br)


@gtscript.function
def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[0, -1, 0] - c * b0[0, -1, 0])


@gtscript.function
def fx1_c_negative(c, bl, b0):
    return (1.0 + c) * (bl + c * b0)


@gtscript.function
def final_flux(c, q, fx1, tmp):
    return q[0, -1, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)


def get_flux(q: FloatField, c: FloatField, al: FloatField):
    from __externals__ import mord

    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[0, 1, 0] - q[0, 0, 0]
    b0 = bl + br

    if mord == 5:
        smt5 = is_smt5_mord5(bl, br)
    else:
        smt5 = is_smt5_most_mords(bl, br, b0)

    if smt5[0, -1, 0] == 0:
        tmp = smt5[0, -1, 0] + smt5
    else:
        tmp = smt5[0, -1, 0]

    fx1 = fx1_fn(c, br, b0, bl)
    return q[0, -1, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


def finalflux_ord8plus(q: FloatField, c: FloatField, bl: FloatField, br: FloatField):
    b0 = bl + br
    fx1 = fx1_fn(c, br, b0, bl)
    return q[0, -1, 0] + fx1 if c > 0.0 else q + fx1


def dm_jord8plus_fcn(q: FloatField):
    xt = 0.25 * (q[0, 1, 0] - q[0, -1, 0])
    dqr = max(max(q, q[0, -1, 0]), q[0, 1, 0]) - q
    dql = q - min(min(q, q[0, -1, 0]), q[0, 1, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@gtstencil()
def dm_jord8plus(q: FloatField, al: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        dm = dm_jord8plus_fcn(q)


def al_jord8plus_fcn(q: FloatField, dm: FloatField):
    return 0.5 * (q[0, -1, 0] + q) + 1.0 / 3.0 * (dm[0, -1, 0] - dm)


@gtstencil()
def al_jord8plus(q: FloatField, al: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        al = al_jord8plus_fcn(q, dm)


def blbr_jord8_fcn(q: FloatField, al: FloatField, dm: FloatField):
    xt = 2.0 * dm
    aldiff = al - q
    aldiffj = al[0, 1, 0] - q
    bl = -1.0 * sign(min(abs(xt), abs(aldiff)), xt)
    br = sign(min(abs(xt), abs(aldiffj)), xt)
    return bl, br


@gtstencil()
def blbr_jord8(
    q: FloatField, al: FloatField, bl: FloatField, br: FloatField, dm: FloatField
):
    with computation(PARALLEL), interval(...):
        bl, br = blbr_jord8_fcn(q, al, dm)


def xt_dya_edge_0_base(q, dya):
    return 0.5 * (
        ((2.0 * dya + dya[0, -1, 0]) * q - dya * q[0, -1, 0]) / (dya[0, -1, 0] + dya)
        + ((2.0 * dya[0, 1, 0] + dya[0, 2, 0]) * q[0, 1, 0] - dya[0, 1, 0] * q[0, 2, 0])
        / (dya[0, 1, 0] + dya[0, 2, 0])
    )


def xt_dya_edge_1_base(q, dya):
    return 0.5 * (
        (
            (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
            - dya[0, -1, 0] * q[0, -2, 0]
        )
        / (dya[0, -2, 0] + dya[0, -1, 0])
        + ((2.0 * dya + dya[0, 1, 0]) * q - dya * q[0, 1, 0]) / (dya + dya[0, 1, 0])
    )


def xt_dya_edge_0(q, dya):
    from __externals__ import xt_minmax

    xt = xt_dya_edge_0_base(q, dya)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        maxq = max(max(max(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        xt = min(max(xt, minq), maxq)
    return xt


def xt_dya_edge_1(q, dya):
    from __externals__ import xt_minmax

    xt = xt_dya_edge_1_base(q, dya)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        maxq = max(max(max(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        xt = min(max(xt, minq), maxq)
    return xt


def south_edge_jord8plus_0(q: FloatField, dya: FloatField, dm: FloatField):
    bl = s14 * dm[0, -1, 0] + s11 * (q[0, -1, 0] - q)
    xt = xt_dya_edge_0(q, dya)
    br = xt - q
    return bl, br


def south_edge_jord8plus_1(q: FloatField, dya: FloatField, dm: FloatField):
    xt = xt_dya_edge_1(q, dya)
    bl = xt - q
    xt = s15 * q + s11 * q[0, 1, 0] - s14 * dm[0, 1, 0]
    br = xt - q
    return bl, br


def south_edge_jord8plus_2(q: FloatField, dm: FloatField, al: FloatField):
    xt = s15 * q[0, -1, 0] + s11 * q - s14 * dm
    bl = xt - q
    br = al[0, 1, 0] - q
    return bl, br


def north_edge_jord8plus_0(q: FloatField, dm: FloatField, al: FloatField):
    bl = al - q
    xt = s15 * q[0, 1, 0] + s11 * q + s14 * dm
    br = xt - q
    return bl, br


def north_edge_jord8plus_1(q: FloatField, dya: FloatField, dm: FloatField):
    xt = s15 * q + s11 * q[0, -1, 0] + s14 * dm[0, -1, 0]
    bl = xt - q
    xt = xt_dya_edge_0(q, dya)
    br = xt - q
    return bl, br


def north_edge_jord8plus_2(q: FloatField, dya: FloatField, dm: FloatField):
    xt = xt_dya_edge_1(q, dya)
    bl = xt - q
    br = s11 * (q[0, 1, 0] - q) - s14 * dm[0, 1, 0]
    return bl, br


@gtscript.function
def pert_ppm_positive_definite_constraint_fcn(
    a0: FloatField, al: FloatField, ar: FloatField
):
    da1 = 0.0
    a4 = 0.0
    fmin = 0.0
    if a0 <= 0.0:
        al = 0.0
        ar = 0.0
    else:
        a4 = -3.0 * (ar + al)
        da1 = ar - al
        if abs(da1) < -a4:
            fmin = a0 + 0.25 / a4 * da1 ** 2 + a4 * (1.0 / 12.0)
            if fmin < 0.0:
                if ar > 0.0 and al > 0.0:
                    ar = 0.0
                    al = 0.0
                elif da1 > 0.0:
                    ar = -2.0 * al
            else:
                al = -2.0 * ar

    return al, ar


@gtstencil()
def pert_ppm_positive_definite_constraint(
    a0: FloatField, al: FloatField, ar: FloatField
):
    with computation(PARALLEL), interval(...):
        al, ar = pert_ppm_positive_definite_constraint_fcn(a0, al, ar)


@gtscript.function
def pert_ppm_standard_constraint_fcn(a0: FloatField, al: FloatField, ar: FloatField):
    da1 = 0.0
    da2 = 0.0
    a6da = 0.0
    if al * ar < 0.0:
        da1 = al - ar
        da2 = da1 ** 2
        a6da = 3.0 * (al + ar) * da1
        if a6da < -da2:
            ar = -2.0 * al
        elif a6da > da2:
            al = -2.0 * ar
    else:
        # effect of dm=0 included here
        al = 0.0
        ar = 0.0
    return al, ar


@gtstencil()
def pert_ppm_standard_constraint(a0: FloatField, al: FloatField, ar: FloatField):
    with computation(PARALLEL), interval(...):
        al, ar = pert_ppm_standard_constraint_fcn(a0, al, ar)


def compute_al(q, dyvar, jord, ifirst, ilast, js1, je3, kstart=0, nk=None):
    if nk is None:
        nk = grid().npz - kstart
    dimensions = q.shape
    local_origin = (origin[0], origin[1], kstart)
    al = utils.make_storage_from_shape(dimensions, local_origin)
    if jord < 8:
        main_al_ord_under8(
            q,
            al,
            origin=(ifirst, js1, kstart),
            domain=(ilast - ifirst + 1, je3 - js1 + 1, nk),
        )
        x_edge_domain = (dimensions[0], 1, nk)
        if not grid().nested and spec.namelist.grid_type < 3:
            # South Edge
            if grid().south_edge:
                al_x_under8_edge_0(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().js - 1, kstart),
                    domain=x_edge_domain,
                )
                al_x_under8_edge_1(
                    q, dyvar, al, origin=(0, grid().js, kstart), domain=x_edge_domain
                )
                al_x_under8_edge_2(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().js + 1, kstart),
                    domain=x_edge_domain,
                )
            # North Edge
            if grid().north_edge:
                al_x_under8_edge_0(
                    q, dyvar, al, origin=(0, grid().je, kstart), domain=x_edge_domain
                )
                al_x_under8_edge_1(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().je + 1, kstart),
                    domain=x_edge_domain,
                )
                al_x_under8_edge_2(
                    q,
                    dyvar,
                    al,
                    origin=(0, grid().je + 2, kstart),
                    domain=x_edge_domain,
                )
        if jord < 0:
            floor_cap(
                al,
                0.0,
                origin=(ifirst, grid().js - 1, kstart),
                domain=(ilast - ifirst + 1, grid().njc + 3, nk),
            )

    return al


def compute_al_fcn(q: FloatField, dya: FloatField):
    from __externals__ import j_end, j_start, jord

    assert __INLINED(jord < 8), "Not implemented"
    # {
    al = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])

    if __INLINED(jord < 0):
        assert __INLINED(False), "Not tested"
        al = max(al, 0.0)

    with horizontal(region[:, j_start - 1], region[:, j_end]):
        al = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q

    with horizontal(region[:, j_start], region[:, j_end + 1]):
        al = 0.5 * (
            (
                (2.0 * dya[0, -1, 0] + dya[0, -2, 0]) * q[0, -1, 0]
                - dya[0, -1, 0] * q[0, -2, 0]
            )
            / (dya[0, -2, 0] + dya[0, -1, 0])
            + (
                (2.0 * dya[0, 0, 0] + dya[0, 1, 0]) * q[0, 0, 0]
                - dya[0, 0, 0] * q[0, 1, 0]
            )
            / (dya[0, 0, 0] + dya[0, 1, 0])
        )

    with horizontal(region[:, j_start + 1], region[:, j_end + 2]):
        al = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]

    # }

    return al


def compute_blbr_ord8plus(q: FloatField, dya: FloatField):
    from __externals__ import j_end, j_start, jord, namelist

    dm = dm_jord8plus_fcn(q)
    al = al_jord8plus_fcn(q, dm)

    assert __INLINED(jord == 8), "Unimplemented jord"
    # {
    bl, br = blbr_jord8_fcn(q, al, dm)
    # }

    assert __INLINED(
        namelist.grid_type < 3
    ), "Remainder of function assumes (grid_type < 3)."
    # {
    with horizontal(region[:, j_start - 1]):
        bl, br = south_edge_jord8plus_0(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_start]):
        bl, br = south_edge_jord8plus_1(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_start + 1]):
        bl, br = south_edge_jord8plus_2(q, dm, al)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end - 1]):
        bl, br = north_edge_jord8plus_0(q, dm, al)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end]):
        bl, br = north_edge_jord8plus_1(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[:, j_end + 1]):
        bl, br = north_edge_jord8plus_2(q, dya, dm)
        bl, br = pert_ppm_standard_constraint_fcn(q, bl, br)
    # }

    return bl, br


# Optimized PPM in perturbation form:
def pert_ppm(a0, al, ar, iv, istart, jstart, kstart, ni, nj, nk):
    r12 = 1.0 / 12.0
    if iv == 0:
        pert_ppm_positive_definite_constraint(
            a0, al, ar, r12, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    else:
        pert_ppm_standard_constraint(
            a0, al, ar, origin=(istart, jstart, kstart), domain=(ni, nj, nk)
        )
    return al, ar


def _compute_flux_stencil(
    q: FloatField, c: FloatField, dya: FloatField, yflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al_fcn(q, dya)
            yflux = get_flux(q, c, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dya)
            yflux = finalflux_ord8plus(q, c, bl, br)


def compute_flux(
    q: FloatField,
    c: FloatField,
    flux: FloatField,
    jord: int,
    ifirst: int,
    ilast: int,
    kstart: int = 0,
    nk: Optional[int] = None,
):
    """
    Compute y-flux using the PPM method.

    Args:
        q (in): Transported scalar
        c (in): Courant number
        flux (out): Flux
        jord: Method selector
        ifirst: Starting index of the I-dir compute domain
        ilast: Final index of the I-dir compute domain
        kstart: First index of the K-dir compute domain
        nk: Number of indices in the K-dir compute domain
    """
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    mord = abs(jord)
    if mord not in [5, 6, 7, 8]:
        raise NotImplementedError(
            f"Unimplemented hord value, {jord}. "
            "Currently only support hord={5, 6, 7, 8}"
        )

    stencil = gtstencil(
        definition=_compute_flux_stencil,
        externals={
            "jord": jord,
            "mord": mord,
            "xt_minmax": True,
        },
    )
    ni = ilast - ifirst + 1
    stencil(
        q,
        c,
        grid.dya,
        flux,
        origin=(ifirst, grid.js, kstart),
        domain=(ni, grid.njc + 1, nk),
    )
