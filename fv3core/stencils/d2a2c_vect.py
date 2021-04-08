from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from fv3core.utils import corners, global_config
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0
BIG_NUMBER = 1.0e30


@gtscript.function
def contravariant(v1, v2, cosa, rsin2):
    """
    Retrieve the contravariant component of the wind from its covariant
    component and the covariant component in the "other" (x/y) direction.

    For an orthogonal grid, cosa would be 0 and rsin2 would be 1, meaning
    the contravariant component is equal to the covariant component.
    However, the gnomonic cubed sphere grid is not orthogonal.

    Args:
        v1: covariant component of the wind for which we want to get the
            contravariant component
        v2: covariant component of the wind for the other direction,
            i.e. y if v1 is in x, x if v1 is in y
        cosa: cosine of the angle between the local x-direction and y-direction.
        rsin2: 1 / (sin(alpha))^2, where alpha is the angle between the local
            x-direction and y-direction

    Returns:
        v1_contravariant: contravariant component of v1
    """
    # From technical docs on FV3 cubed sphere grid:
    # The gnomonic cubed sphere grid is not orthogonal, meaning
    # the u and v vectors have some overlapping component. We can decompose
    # the total wind U in two ways, as a linear combination of the
    # coordinate vectors ("contravariant"):
    #    U = u_contravariant * u_dir + v_contravariant * v_dir
    # or as the projection of the vector onto the coordinate
    #    u_covariant = U dot u_dir
    #    v_covariant = U dot v_dir
    # The names come from the fact that the covariant vectors vary
    # (under a change in coordinate system) the same way the coordinate values do,
    # while the contravariant vectors vary in the "opposite" way.
    #
    # equations from FV3 technical documentation
    # u_cov = u_contra + v_contra * cos(alpha)  (eq 3.4)
    # v_cov = u_contra * cos(alpha) + v_contra  (eq 3.5)
    #
    # u_contra = u_cov - v_contra * cos(alpha)  (1, from 3.4)
    # v_contra = v_cov - u_contra * cos(alpha)  (2, from 3.5)
    # u_contra = u_cov - (v_cov - u_contra * cos(alpha)) * cos(alpha)  (from 1 & 2)
    # u_contra = u_cov - v_cov * cos(alpha) + u_contra * cos2(alpha) (follows)
    # u_contra * (1 - cos2(alpha)) = u_cov - v_cov * cos(alpha)
    # u_contra = u_cov/(1 - cos2(alpha)) - v_cov * cos(alpha)/(1 - cos2(alpha))
    # matches because rsin = 1 /(1 + cos2(alpha)),
    #                 cosa*rsin = cos(alpha)/(1 + cos2(alpha))

    # recall that:
    # rsin2 is 1/(sin(alpha))^2
    # cosa is cos(alpha)

    return (v1 - v2 * cosa) * rsin2


@gtscript.function
def vol_conserv_cubic_interp_func_x(u):
    return c1 * u[-2, 0, 0] + c2 * u[-1, 0, 0] + c3 * u


@gtscript.function
def vol_conserv_cubic_interp_func_x_rev(u):
    return c1 * u[1, 0, 0] + c2 * u + c3 * u[-1, 0, 0]


@gtscript.function
def vol_conserv_cubic_interp_func_y(v):
    return c1 * v[0, -2, 0] + c2 * v[0, -1, 0] + c3 * v


@gtscript.function
def vol_conserv_cubic_interp_func_y_rev(v):
    return c1 * v[0, 1, 0] + c2 * v + c3 * v[0, -1, 0]


@gtscript.function
def lagrange_y_func_p1(qx):
    return a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])


@gtstencil()
def lagrange_interpolation_y_p1(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func_p1(qx)


@gtscript.function
def lagrange_x_func_p1(qy):
    return a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])


@gtstencil()
def lagrange_interpolation_x_p1(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func_p1(qy)


@gtscript.function
def interp_winds_d_to_a(u, v):
    """
    Interpolate winds from the d-grid to a-grid.

    This uses a Lagrange interpolation on the interior and averaging on the
    boundaries. D2A2C_AVG_OFFSET is an external that describes how far the
    averaging should go before switching to Lagrangian interpolation. For
    sufficiently small grids, this should be set to -1, otherwise 3. Note that
    this makes the stencil code in d2a2c grid-dependent!
    """
    from __externals__ import (
        D2A2C_AVG_OFFSET,
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
    )

    utmp = BIG_NUMBER
    vtmp = BIG_NUMBER

    with horizontal(region[:, local_js - 1 : local_je + 2]):
        utmp = lagrange_y_func_p1(u)
    with horizontal(region[local_is - 1 : local_ie + 2, :]):
        vtmp = lagrange_x_func_p1(v)

    # WARNING: This introduces grid-size dependence into the stencil code.
    with horizontal(
        region[:, : j_start + D2A2C_AVG_OFFSET],
        region[:, j_end - D2A2C_AVG_OFFSET + 1 :],
        region[: i_start + D2A2C_AVG_OFFSET, :],
        region[i_end - D2A2C_AVG_OFFSET + 1 :, :],
    ):
        utmp = 0.5 * (u + u[0, 1, 0])
        vtmp = 0.5 * (v + v[1, 0, 0])

    return utmp, vtmp


@gtscript.function
def edge_interpolate4_x(ua, dxa):
    t1 = dxa[-2, 0] + dxa[-1, 0]
    t2 = dxa[0, 0] + dxa[1, 0]
    n1 = (t1 + dxa[-1, 0]) * ua[-1, 0, 0] - dxa[-1, 0] * ua[-2, 0, 0]
    n2 = (t1 + dxa[0, 0]) * ua[0, 0, 0] - dxa[0, 0] * ua[1, 0, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


@gtscript.function
def edge_interpolate4_y(va, dya):
    t1 = dya[0, -2] + dya[0, -1]
    t2 = dya[0, 0] + dya[0, 1]
    n1 = (t1 + dya[0, -1]) * va[0, -1, 0] - dya[0, -1] * va[0, -2, 0]
    n2 = (t1 + dya[0, 0]) * va[0, 0, 0] - dya[0, 0] * va[0, 1, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


def _d2a2c_vect(
    cosa_s: FloatFieldIJ,
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    rsin2: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    u: FloatField,
    ua: FloatField,
    uc: FloatField,
    utc: FloatField,
    v: FloatField,
    va: FloatField,
    vc: FloatField,
    vtc: FloatField,
):
    from __externals__ import (
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
    )

    with computation(PARALLEL), interval(...):

        utmp, vtmp = interp_winds_d_to_a(u, v)

        with horizontal(
            region[local_is - 2 : local_ie + 3, local_js - 2 : local_je + 3]
        ):
            ua = contravariant(utmp, vtmp, cosa_s, rsin2)
            va = contravariant(vtmp, utmp, cosa_s, rsin2)

        # A -> C
        # Fix the edges
        utmp = corners.fill_corners_3cells_mult_x(
            utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        ua = corners.fill_corners_2cells_mult_x(
            ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )

        # X

        with horizontal(
            region[local_is - 1 : local_ie + 3, local_js - 1 : local_je + 2]
        ):
            uc = lagrange_x_func(utmp)
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # West
        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with horizontal(region[i_start, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # East
        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with horizontal(region[i_end + 1, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # Fill corners for Y

        vtmp = corners.fill_corners_3cells_mult_y(
            vtmp, utmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        va = corners.fill_corners_2cells_mult_y(
            va, ua, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )

        # Y

        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 3]
        ):
            vc = lagrange_y_func(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start - 1]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_start + 1]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 1]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 2]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

    # When this is a function inside c_sw, use the return statement below
    # return uc, vc, ua, va, utc, vtc


def compute(
    u: FloatField,
    ua: FloatField,
    uc: FloatField,
    utc: FloatField,
    v: FloatField,
    va: FloatField,
    vc: FloatField,
    vtc: FloatField,
):
    """
    D2A2C. R2D2. <Insert Other Star War Reference>.

    Args:
        u: ???
        ua: ???
        uc: ???
        utc: ???
        v: ???
        va: ???
        vc: ???
        vtc: ???

    Grid variables referenced:
        cosa_s, cosa_u, cosa_v, dxa, dya, rsin2, rsin_u, rsin_v,
        sin_sg1, sin_sg2, sin_sg3, sin_sg4.
    """
    grid = spec.grid
    namelist = spec.namelist
    assert namelist.grid_type < 3
    origin = grid.compute_origin(add=(-2, -2, 0))
    domain = grid.domain_shape_compute(add=(4, 4, 0))
    grid_args = {
        "cosa_s": grid.cosa_s,
        "cosa_u": grid.cosa_u,
        "cosa_v": grid.cosa_v,
        "dxa": grid.dxa,
        "dya": grid.dya,
        "rsin2": grid.rsin2,
        "rsin_u": grid.rsin_u,
        "rsin_v": grid.rsin_v,
        "sin_sg1": grid.sin_sg1,
        "sin_sg2": grid.sin_sg2,
        "sin_sg3": grid.sin_sg3,
        "sin_sg4": grid.sin_sg4,
    }
    state_args = {
        "u": u,
        "ua": ua,
        "uc": uc,
        "utc": utc,
        "v": v,
        "va": va,
        "vc": vc,
        "vtc": vtc,
    }

    ax_offsets = axis_offsets(spec.grid, origin, domain)

    _c12_stencil = gtscript.stencil(
        definition=_d2a2c_vect,
        externals={"D2A2C_AVG_OFFSET": -1, **ax_offsets},
        backend=global_config.get_backend(),
        rebuild=global_config.get_rebuild(),
    )

    _default_stencil = gtscript.stencil(
        definition=_d2a2c_vect,
        externals={"D2A2C_AVG_OFFSET": 3, **ax_offsets},
        backend=global_config.get_backend(),
        rebuild=global_config.get_rebuild(),
    )

    if namelist.npx <= 13 and namelist.layout[0] > 1:
        stencil = _c12_stencil
    else:
        stencil = _default_stencil

    stencil(
        **grid_args,
        **state_args,
        origin=origin,
        domain=domain,
    )
