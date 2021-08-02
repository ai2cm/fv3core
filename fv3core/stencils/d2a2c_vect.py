import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from fv3core.utils.typing import FloatField, FloatFieldIJ


c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0
OFFSET = 2


def grid():
    return spec.grid


def set_tmps(utmp: FloatField, vtmp: FloatField, big_number: float):
    with computation(PARALLEL), interval(...):
        utmp = big_number
        vtmp = big_number


# almost the same as a2b_ord4's version
@gtscript.function
def lagrange_y_func_p1(qx):
    return a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])


def lagrange_interpolation_y_p1(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func_p1(qx)


@gtscript.function
def lagrange_x_func_p1(qy):
    return a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])


def lagrange_interpolation_x_p1(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func_p1(qy)


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


def contravariant_stencil(
    u: FloatField,
    v: FloatField,
    cosa: FloatFieldIJ,
    rsin: FloatFieldIJ,
    out: FloatField,
):
    with computation(PARALLEL), interval(...):
        out = contravariant(u, v, cosa, rsin)


def contravariant_components(
    utmp: FloatField,
    vtmp: FloatField,
    cosa_s: FloatFieldIJ,
    rsin2: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        ua = contravariant(utmp, vtmp, cosa_s, rsin2)
        va = contravariant(vtmp, utmp, cosa_s, rsin2)


def ut_main(
    utmp: FloatField,
    uc: FloatField,
    v: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    ut: FloatField,
):
    with computation(PARALLEL), interval(...):
        uc = lagrange_x_func(utmp)
        ut = contravariant(uc, v, cosa_u, rsin_u)


def vt_main(
    vtmp: FloatField,
    vc: FloatField,
    u: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    vt: FloatField,
):
    with computation(PARALLEL), interval(...):
        vc = lagrange_y_func(vtmp)
        vt = contravariant(vc, u, cosa_v, rsin_v)


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


def vol_conserv_cubic_interp_x(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x(utmp)


def vol_conserv_cubic_interp_x_rev(utmp: FloatField, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x_rev(utmp)


def vol_conserv_cubic_interp_y(vtmp: FloatField, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc = vol_conserv_cubic_interp_func_y(vtmp)


class DGrid2AGrid2CGridVectors:
    """
    Fortran name d2a2c_vect
    """

    def __init__(self, grid, namelist, dord4):
        if namelist.grid_type >= 3:
            raise Exception("unimplemented grid_type >= 3")
        self.grid = grid
        self._big_number = 1e30  # 1e8 if 32 bit
        self._nx = self.grid.ie + 1  # grid.npx + 2
        self._ny = self.grid.je + 1  # grid.npy + 2
        self._i1 = self.grid.is_ - 1
        self._j1 = self.grid.js - 1
        id_ = 1 if dord4 else 0
        pad = 2 + 2 * id_
        npt = 4 if not self.grid.nested else 0
        if npt > self.grid.nic - 1 or npt > self.grid.njc - 1:
            npt = 0
        self._utmp = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)),
            self.grid.full_origin(),
        )
        self._vtmp = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)), self.grid.full_origin()
        )

        js1 = self.grid.js - 1
        je1 = self.grid.je + 1
        is1 = self.grid.isd
        ie1 = self.grid.ied

        is2 = self.grid.is_ - 1
        ie2 = self.grid.ie + 1
        js2 = self.grid.jsd
        je2 = self.grid.jed

        ifirst = self.grid.is_ - 1
        ilast = self.grid.ie + 2
        idiff = ilast - ifirst + 1

        jfirst = self.grid.js - 1
        jlast = self.grid.je + 2
        jdiff = jlast - jfirst + 1

        js3 = self.grid.jsd
        je3 = self.grid.jed
        jdiff3 = je3 - js3 + 1

        self._set_tmps = FrozenStencil(
            func=set_tmps,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )

        self._lagrange_interpolation_y_p1 = FrozenStencil(
            func=lagrange_interpolation_y_p1,
            origin=(is1, js1, 0),
            domain=(ie1 - is1 + 1, je1 - js1 + 1, self.grid.npz),
        )

        self._lagrange_interpolation_x_p1 = FrozenStencil(
            func=lagrange_interpolation_x_p1,
            origin=(is2, js2, 0),
            domain=(ie2 - is2 + 1, je2 - js2 + 1, self.grid.npz),
        )

        origin = self.grid.full_origin()
        domain = self.grid.domain_shape_full()

        if namelist.npx <= 13 and namelist.layout[0] > 1:
            d2a2c_avg_offset = -1
        else:
            d2a2c_avg_offset = 3
        self._contravariant_components = FrozenStencil(
            func=contravariant_components,
            origin=(self.grid.is_ - 1 - id_, self.grid.js - 1 - id_, 0),
            domain=(self.grid.nic + pad, self.grid.njc + pad, self.grid.npz),
        )

        self._ut_main = FrozenStencil(
            func=ut_main,
            origin=(ifirst, self._j1, 0),
            domain=(idiff, self.grid.njc + 2, self.grid.npz),
        )

        # Ydir:
        self._vt_main = FrozenStencil(
            func=vt_main,
            origin=(self._i1, jfirst, 0),
            domain=(self.grid.nic + 2, jdiff, self.grid.npz),
        )

    def __call__(self, uc, vc, u, v, ua, va, utc, vtc):
        """
        Calculate velocity vector from D-grid to A-grid to C-grid.

        Args:
            uc: C-grid x-velocity (inout)
            vc: C-grid y-velocity (inout)
            u: D-grid x-velocity (in)
            v: D-grid y-velocity (in)
            ua: A-grid x-velocity (inout)
            va: A-grid y-velocity (inout)
            utc: C-grid u * dx (inout)
            vtc: C-grid v * dy (inout)
        """
        self._set_tmps(
            self._utmp,
            self._vtmp,
            self._big_number,
        )

        self._lagrange_interpolation_y_p1(
            u,
            self._utmp,
        )

        self._lagrange_interpolation_x_p1(
            v,
            self._vtmp,
        )

        # contra-variant components at cell center
        self._contravariant_components(
            self._utmp,
            self._vtmp,
            self.grid.cosa_s,
            self.grid.rsin2,
            ua,
            va,
        )

        self._ut_main(
            self._utmp,
            uc,
            v,
            self.grid.cosa_u,
            self.grid.rsin_u,
            utc,
        )

        self._vt_main(
            self._vtmp,
            vc,
            u,
            self.grid.cosa_v,
            self.grid.rsin_v,
            vtc,
        )
