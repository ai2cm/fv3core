import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.cs_limiters as cs_limiters

sd = utils.sd


def grid():
    return spec.grid


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value


@utils.stencil()
def set_vals_0(gam: sd, q: sd, delp: sd, a4_1: sd, qs: sd):
    with computation(PARALLEL):
        with interval(0, 2):
            # set top
            gam = 0.5
            q = 1.5 * a4_1
    with computation(FORWARD):
        with interval(1, -1):
            # set middle
            grid_ratio = delp[0, 0, -1] / delp
            bet = 2.0 + grid_ratio + grid_ratio - gam
            q = (3.0 * (a4_1[0, 0, -1] + a4_1[0, 0, 0]) - q[0, 0, -1]) / bet
            gam[0, 0, +1] = grid_ratio / bet
    with computation(PARALLEL):
        with interval(-2, -1):
            # set bottom
            grid_ratio = delp[0, 0, -1] / delp
            q = (
                3.0 * (a4_1[0, 0, -1] + a4_1[0, 0, 0]) - (qs * grid_ratio) - q[0, 0, -1]
            ) / (2 + grid_ratio + grid_ratio - gam)
            q[0, 0, 1] = qs
    with computation(BACKWARD), interval(0, -2):
        q = q - (gam[0, 0, 1] * q[0, 0, 1])


@utils.stencil()
def set_vals_1(gam: sd, q: sd, delp: sd, a4_1: sd, qs: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            # set top
            grid_ratio = delp[0, 0, 1] / delp
            bet = grid_ratio * (grid_ratio + 0.5)
            q = (
                (grid_ratio + grid_ratio) * (grid_ratio + 1.0) * a4_1 + a4_1[0, 0, 1]
            ) / bet
            gam = (1.0 + grid_ratio * (grid_ratio + 1.5)) / bet
    with computation(FORWARD):
        with interval(1, -1):
            # set middle
            d4 = delp[0, 0, -1] / delp
            bet = 2.0 + d4 + d4 - gam[0, 0, -1]
            q = (3.0 * (a4_1[0, 0, -1] + d4 * a4_1) - q[0, 0, -1]) / bet
            gam = d4 / bet
    with computation(PARALLEL):
        with interval(-1, None):
            # set bottom
            a_bot = 1.0 + (delp[0, 0, -2] / delp[0, 0, -1]) * (
                (delp[0, 0, -2] / delp[0, 0, -1]) + 1.5
            )
            q = (
                2.0
                * (delp[0, 0, -2] / delp[0, 0, -1])
                * ((delp[0, 0, -2] / delp[0, 0, -1]) + 1.0)
                * a4_1[0, 0, -1]
                + a4_1[0, 0, -2]
                - a_bot * q[0, 0, -1]
            ) / (
                (delp[0, 0, -2] / delp[0, 0, -1])
                * ((delp[0, 0, -2] / delp[0, 0, -1]) + 0.5)
                - a_bot * gam[0, 0, -1]
            )

    with computation(BACKWARD), interval(0, -1):
        q = q - (gam[0, 0, 0] * q[0, 0, 1])


@utils.stencil()
def set_avals(q: sd, a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        a4_2 = q
        a4_3 = q[0, 0, 1]
        a4_4 = 3.0 * (2.0 * a4_1 - (q + q[0, 0, 1]))


@utils.stencil()
def Apply_constraints(q: sd, gam: sd, a4_1: sd, a4_2: sd, a4_3: sd, iv: float):
    with computation(PARALLEL):
        with interval(1, 1):
            q = (
                q
                if (q < a4_1[0, 0, -1] or q < a4_1)
                else a4_1[0, 0, -1]
                if a4_1[0, 0, -1] > a4_1
                else a4_1
            )
            q = (
                q
                if (q > a4_1[0, 0, -1] or q > a4_1)
                else a4_1[0, 0, -1]
                if a4_1[0, 0, -1] < a4_1
                else a4_1
            )
        with interval(1, -1):
            # do top
            gam = a4_1 - a4_1[0, 0, -1]
        with interval(2, -2):
            # do middle
            if gam[0, 0, -1] * gam[0, 0, 1] > 0:
                q = (
                    q
                    if (q < a4_1[0, 0, -1] or q < a4_1)
                    else a4_1[0, 0, -1]
                    if a4_1[0, 0, -1] > a4_1
                    else a4_1
                )
                q = (
                    q
                    if (q > a4_1[0, 0, -1] or q > a4_1)
                    else a4_1[0, 0, -1]
                    if a4_1[0, 0, -1] < a4_1
                    else a4_1
                )
            elif gam[0, 0, -1] > 0:
                q = (
                    q
                    if (q > a4_1[0, 0, -1] or q > a4_1)
                    else a4_1[0, 0, -1]
                    if a4_1[0, 0, -1] < a4_1
                    else a4_1
                )
            else:
                q = (
                    q
                    if (q < a4_1[0, 0, -1] or q < a4_1)
                    else a4_1[0, 0, -1]
                    if a4_1[0, 0, -1] > a4_1
                    else a4_1
                )
                if iv == 0:
                    q = q if q > 0.0 else 0.0
        with interval(-2, -1):
            q = (
                q
                if (q < a4_1[0, 0, -1] or q < a4_1)
                else a4_1[0, 0, -1]
                if a4_1[0, 0, -1] > a4_1
                else a4_1
            )
            q = (
                q
                if (q > a4_1[0, 0, -1] or q > a4_1)
                else a4_1[0, 0, -1]
                if a4_1[0, 0, -1] < a4_1
                else a4_1
            )
        with interval(0, -1):
            # re-set a4_2 and a4_3
            a4_2 = q
            a4_3 = q[0, 0, 1]


@utils.stencil()
def set_extm(extm: sd, a4_1: sd, a4_2: sd, a4_3: sd, gam: sd):
    with computation(PARALLEL):
        with interval(0, 1):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0
        with interval(-1, None):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.0
        with interval(1, -1):
            extm - gam * gam[0, 0, 1] < 0.0


@utils.stencil()
def set_exts(a4_4: sd, ext5: sd, ext6: sd, a4_1: sd, a4_2: sd, a4_3: sd):
    with computation(PARALLEL), interval(...):
        x0 = 2.0 * a4_1 - (a4_2 + a4_3)
        x1 = absolute_value(a4_2 - a4_3)
        a4_4 = 3.0 * x0
        ext5 = absolute_value(x0) > x1
        ext6 = absolute_value(a4_4) > x1


@utils.stencil()
def set_top_as(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, iv: int):
    with computation(PARALLEL):
        with interval(0, 1):
            if iv == 0:
                a4_2 = a4_2 if a4_2 > 0.0 else 0.0
            elif iv == -1:
                if a4_2 * a4_1 <= 0.0:
                    a4_2 = 0.0
            elif iv == 2:
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
            if iv != 2:
                a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))
        with interval(1, None):
            a4_4 = 3 * (2 * a4_1 - (a4_2 + a4_3))


@utils.stencil()
def set_inner_as(
    a4_1: sd,
    a4_2: sd,
    a4_3: sd,
    a4_4: sd,
    gam: sd,
    extm: sd,
    ext5: sd,
    ext6: sd,
    kord: int,
):
    with computation(PARALLEL), interval(...):
        if abs(kord) < 9:
            # left edges?
            pmp_1 = a4_1 - gam[0, 0, 1]
            lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
            tmp_min = (
                a4_1
                if (a4_1 < pmp_1) and (a4_1 < lac_1)
                else pmp_1
                if pmp_1 < lac_1
                else lac_1
            )
            tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
            tmp_max = (
                a4_1
                if (a4_1 > pmp_1) and (a4_1 > lac_1)
                else pmp_1
                if pmp_1 > lac_1
                else lac_1
            )
            a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
            # right edges?
            pmp_2 = a4_1 + 2.0 * gam[0, 0, 1]
            lac_2 = pmp_2 + 1.5 * gam[0, 0, -1]
            tmp_min = (
                a4_1
                if (a4_1 < pmp_2) and (a4_1 < lac_2)
                else pmp_2
                if pmp_2 < lac_2
                else lac_2
            )
            tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
            tmp_max = (
                a4_1
                if (a4_1 > pmp_2) and (a4_1 > lac_2)
                else pmp_2
                if pmp_2 > lac_2
                else lac_2
            )
            a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))

        elif abs(kord) == 9:
            if extm and extm[0, 0, -1]:
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
            elif extm and extm[0, 0, 1]:
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
            else:
                a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)
                if absolute_value(a4_4) > absolute_value(a4_2 - a4_3):
                    pmp_1 = a4_1 - 2.0 * gam[0, 0, 1]
                    lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
                    tmp_min = (
                        a4_1
                        if (a4_1 < pmp_1) and (a4_1 < lac_1)
                        else pmp_1
                        if pmp_1 < lac_1
                        else lac_1
                    )
                    tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                    tmp_max = (
                        a4_1
                        if (a4_1 > pmp_1) and (a4_1 > lac_1)
                        else pmp_1
                        if pmp_1 > lac_1
                        else lac_1
                    )
                    a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                    pmp_2 = a4_1 + 2.0 * gam[0, 0, 1]
                    lac_2 = pmp_2 + 1.5 * gam[0, 0, -1]
                    tmp_min = (
                        a4_1
                        if (a4_1 < pmp_2) and (a4_1 < lac_2)
                        else pmp_2
                        if pmp_2 < lac_2
                        else lac_2
                    )
                    tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                    tmp_max = (
                        a4_1
                        if (a4_1 > pmp_2) and (a4_1 > lac_2)
                        else pmp_2
                        if pmp_2 > lac_2
                        else lac_2
                    )
                    a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                    a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)

        elif abs(kord) == 10:
            if ext5:
                if ext5[0, 0, -1] or ext5[0, 0, 1]:
                    a4_2 = a4_1
                    a4_3 = a4_1
                elif ext6[0, 0, -1] or ext6[0, 0, 1]:
                    pmp_1 = a4_1 - 2.0 * gam[0, 0, 1]
                    lac_1 = pmp_1 + 1.5 * gam[0, 0, 2]
                    tmp_min = (
                        a4_1
                        if (a4_1 < pmp_1) and (a4_1 < lac_1)
                        else pmp_1
                        if pmp_1 < lac_1
                        else lac_1
                    )
                    tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                    tmp_max = (
                        a4_1
                        if (a4_1 > pmp_1) and (a4_1 > lac_1)
                        else pmp_1
                        if pmp_1 > lac_1
                        else lac_1
                    )
                    a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                    pmp_2 = a4_1 + 2.0 * gam[0, 0, 1]
                    lac_2 = pmp_2 + 1.5 * gam[0, 0, -1]
                    tmp_min = (
                        a4_1
                        if (a4_1 < pmp_2) and (a4_1 < lac_2)
                        else pmp_2
                        if pmp_2 < lac_2
                        else lac_2
                    )
                    tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                    tmp_max = (
                        a4_1
                        if (a4_1 > pmp_2) and (a4_1 > lac_2)
                        else pmp_2
                        if pmp_2 > lac_2
                        else lac_2
                    )
                    a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
            a4_4 = 6.0 * a4_1 - 3.0 * (a4_2 + a4_3)
        else:
            print("kord {0} not implemented yet. Go bug a dev for it.".format(kord))
            pass


@utils.stencil()
def set_bottom_as(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, iv: int):
    with computation(PARALLEL):
        with interval(1, None):
            if iv == 0.0:
                a4_3 = a4_3 if a4_3 > 0.0 else 0.0
            elif iv == -1:
                if a4_3 * a4_1 <= 0.0:
                    a4_3 = 0.0
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


def compute(qs, a4_1, a4_2, a4_3, a4_4, delp, km, i1, i2, iv, kord):
    # TODO: how do we handle 2d-stencils/take a 2d slice of a 3d array?
    # TODO: how do we handle loopy stencils, e.g. q(i,k) = q(i,k) - gam(i,k+1)*q(i,k+1)?
    # Or q(i,k) = (3.*(a4(1,i,k-1)+a4(1,i,k)) - q(i,k-1))/bet?
    # qs is 1-d
    # delp is 2d
    # a4 is 3d but weirdly shaped, probably only care about i=1?
    # TODO how put these all together??
    i_extent = i2 - i1 + 1
    grid = spec.grid
    orig = (i1, grid.js, 0)
    full_orig = (grid.is_, grid.js, 0)
    dom = (i_extent, 1, km)
    ext_dom = (i_extent, 1, km + 1)
    extend_shape = list(delp.shape)
    extend_shape[-1] += 1
    gam = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    q = utils.make_storage_from_shape(tuple(extend_shape), origin=full_orig)

    extm = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext5 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext6 = utils.make_storage_from_shape(delp.shape, origin=full_orig)

    if iv == 2:
        set_vals_0(gam, q, delp, a4_1, qs, origin=orig, domain=ext_dom)
    else:
        set_vals_0(gam, q, delp, a4_1, qs, origin=orig, domain=ext_dom)

    if abs(kord) > 16:
        set_avals(q, a4_1, a4_2, a4_3, a4_4, origin=orig, domain=dom)

    Apply_constraints(q, gam, a4_1, a4_2, a4_3, iv)
    set_extm(extm, a4_1, a4_2, a4_3, gam, origin=orig, domain=dom)

    if abs(kord) > 9:
        set_exts(a4_4, ext5, ext6, a4_1, a4_2, a4_3, origin=orig, domain=dom)

    set_top_as(a4_1, a4_2, a4_3, a4_4, iv, origin=orig, domain=(i_extent, 1, 2))
    set_inner_as(
        a4_1,
        a4_2,
        a4_3,
        a4_4,
        gam,
        extm,
        ext5,
        ext6,
        kord,
        origin=(i1, grid.js, 2),
        domain=(i_extent, 1, km - 4),
    )
    set_bottom_as(
        a4_1,
        a4_2,
        a4_3,
        a4_4,
        extm,
        iv,
        origin=(i1, grid.js, km - 1),
        domain=(i_extent, 1, 2),
    )

    if iv != 2:
        a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(
            a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1
        )
    a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(
        a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, 1, 1
    )
    if iv == 0:
        a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(
            a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 2, km - 4
        )
    a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(
        a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, km - 3, 1
    )
    a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(
        a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, km - 3, 1
    )

    return a4_1, a4_2, a4_3, a4_4
