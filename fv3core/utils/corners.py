from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField


class CopyCorners:
    """
    Helper-class to copy corners corresponding to the fortran functions
    copy_corners_x or copy_corners_y respectively
    """

    def __init__(self, direction: str, temporary_field=None, origin=None, domain=None) -> None:
        self.grid = spec.grid
        if origin is None:
            origin = self.grid.full_origin()
       
        if domain is None:
            domain = self.grid.domain_shape_full(add=(0, 0, 1))
        self._origin=origin
        self._domain=domain
        self.direction = direction
       
    def __call__(self, field: FloatField):
        """
        Fills cell quantity field using corners from itself and multipliers
        in the dirction specified initialization of the instance of this class.
        """
        copy_corners(field, self.direction, self.grid, kslice=slice(self._origin[2], self._origin[0] + self._domain[2]))
       


class FillCornersBGrid:
    """
    Helper-class to fill corners corresponding to the fortran function
    fill_corners with BGRID=.true. and either FILL=YDir or FILL=YDIR
    """

    def __init__(
        self, direction: str, temporary_field=None, origin=None, domain=None
    ) -> None:
        self.grid = spec.grid
        """The grid for this stencil"""
        if origin is None:
            origin = self.grid.full_origin()
        """The origin for the corner computation"""
        if domain is None:
            domain = self.grid.domain_shape_full()
        """The full domain required to do corner computation everywhere"""
        
        self._direction = direction

        

    def __call__(self, field: FloatField):
        fill_corners_2d(field, self.grid, "B", self._direction)



def corner_ke(
    ke: FloatField,
    u: FloatField,
    v: FloatField,
    ut: FloatField,
    vt: FloatField,
    dt: float,
    io1: int,
    jo1: int,
    io2: int,
    vsign: int,
):
    with computation(PARALLEL), interval(...):
        dt6 = dt / 6.0
        
        ke = dt6 * (
            (ut[0, 0, 0] + ut[0, -1, 0]) * ((io1 + 1) * u[0, 0, 0] - (io1 * u[-1, 0, 0]))
            + (vt[0, 0, 0] + vt[-1, 0, 0]) * ((jo1 + 1) * v[0, 0, 0] - (jo1 * v[0, -1, 0]))
            + (
                ((jo1 + 1) * ut[0, 0, 0] - (jo1 * ut[0, -1, 0]))
                + vsign * ((io1 + 1) * vt[0, 0, 0] - (io1 * vt[-1, 0, 0]))
            )
            * ((io2 + 1) * u[0, 0, 0] - (io2 * u[-1, 0, 0]))
        )
        
# FOR DACE
def fill_4corners(q, direction, grid):
    if direction == "x":
        for k in range(q.shape[2]):
            if grid.sw_corner:
                q[grid.is_ - 2, grid.js - 1, k] = q[grid.is_ - 1, grid.js + 1, k]
                q[grid.is_ - 1, grid.js - 1, k] = q[grid.is_ - 1, grid.js, k]
            if grid.se_corner:
                q[grid.ie + 2, grid.js - 1, k] = q[grid.ie + 1, grid.js + 1, k]
                q[grid.ie + 1, grid.js - 1, k] = q[grid.ie + 1, grid.js, k]
            if grid.nw_corner:
                q[grid.is_ - 1, grid.je + 1, k] = q[grid.is_ - 1, grid.je, k]
                q[grid.is_ - 2, grid.je + 1, k] = q[grid.is_ - 1, grid.je - 1, k]
            if grid.ne_corner:
                q[grid.ie + 1, grid.je + 1, k] = q[grid.ie + 1, grid.je, k]
                q[grid.ie + 2, grid.je + 1, k] = q[grid.ie + 1, grid.je - 1, k]
    elif direction == "y":
        for k in range(q.shape[2]):
            if grid.sw_corner:
                q[grid.is_ - 1, grid.js - 1, k] = q[grid.is_, grid.js - 1, k]
                q[grid.is_ - 1, grid.js - 2, k] = q[grid.is_ + 1, grid.js - 1, k]
            if grid.se_corner:
                q[grid.ie + 1, grid.js - 1, k] = q[grid.ie, grid.js - 1, k]
                q[grid.ie + 1, grid.js - 2, k] = q[grid.ie - 1, grid.js - 1, k]
            if grid.nw_corner:
                q[grid.is_ - 1, grid.je + 1, k] = q[grid.is_, grid.je + 1, k]
                q[grid.is_ - 1, grid.je + 2, k] = q[grid.is_ + 1, grid.je + 1, k]
            if grid.ne_corner:
                q[grid.ie + 1, grid.je + 1, k] = q[grid.ie, grid.je + 1, k]
                q[grid.ie + 1, grid.je + 2, k] = q[grid.ie - 1, grid.je + 1, k]
    else:
        raise ValueError("Direction not recognized. Specify either x or y")


def fill2_4corners(q1, q2, direction, grid):
    if direction == "x":
        for k in range(q1.shape[2]):
            if grid.sw_corner:
                q1[grid.is_ - 2, grid.js - 1, k] = q1[grid.is_ - 1, grid.js + 1, k]
                q1[grid.is_ - 1, grid.js - 1, k] = q1[grid.is_ - 1, grid.js, k]
                q2[grid.is_ - 2, grid.js - 1, k] = q2[grid.is_ - 1, grid.js + 1, k]
                q2[grid.is_ - 1, grid.js - 1, k] = q2[grid.is_ - 1, grid.js, k]
            if grid.se_corner:
                q1[grid.ie + 2, grid.js - 1, k] = q1[grid.ie + 1, grid.js + 1, k]
                q1[grid.ie + 1, grid.js - 1, k] = q1[grid.ie + 1, grid.js, k]
                q2[grid.ie + 2, grid.js - 1, k] = q2[grid.ie + 1, grid.js + 1, k]
                q2[grid.ie + 1, grid.js - 1, k] = q2[grid.ie + 1, grid.js, k]
            if grid.nw_corner:
                q1[grid.is_ - 1, grid.je + 1, k] = q1[grid.is_ - 1, grid.je, k]
                q1[grid.is_ - 2, grid.je + 1, k] = q1[grid.is_ - 1, grid.je - 1, k]
                q2[grid.is_ - 1, grid.je + 1, k] = q2[grid.is_ - 1, grid.je, k]
                q2[grid.is_ - 2, grid.je + 1, k] = q2[grid.is_ - 1, grid.je - 1, k]
            if grid.ne_corner:
                q1[grid.ie + 1, grid.je + 1, k] = q1[grid.ie + 1, grid.je, k]
                q1[grid.ie + 2, grid.je + 1, k] = q1[grid.ie + 1, grid.je - 1, k]
                q2[grid.ie + 1, grid.je + 1, k] = q2[grid.ie + 1, grid.je, k]
                q2[grid.ie + 2, grid.je + 1, k] = q2[grid.ie + 1, grid.je - 1, k]
    elif direction == "y":
        for k in range(q1.shape[2]):
            if grid.sw_corner:
                q1[grid.is_ - 1, grid.js - 1, k] = q1[grid.is_, grid.js - 1, k]
                q1[grid.is_ - 1, grid.js - 2, k] = q1[grid.is_ + 1, grid.js - 1, k]
                q2[grid.is_ - 1, grid.js - 1, k] = q2[grid.is_, grid.js - 1, k]
                q2[grid.is_ - 1, grid.js - 2, k] = q2[grid.is_ + 1, grid.js - 1, k]
            if grid.se_corner:
                q1[grid.ie + 1, grid.js - 1, k] = q1[grid.ie, grid.js - 1, k]
                q1[grid.ie + 1, grid.js - 2, k] = q1[grid.ie - 1, grid.js - 1, k]
                q2[grid.ie + 1, grid.js - 1, k] = q2[grid.ie, grid.js - 1, k]
                q2[grid.ie + 1, grid.js - 2, k] = q2[grid.ie - 1, grid.js - 1, k]
            if grid.nw_corner:
                q1[grid.is_ - 1, grid.je + 1, k] = q1[grid.is_, grid.je + 1, k]
                q1[grid.is_ - 1, grid.je + 2, k] = q1[grid.is_ + 1, grid.je + 1, k]
                q2[grid.is_ - 1, grid.je + 1, k] = q2[grid.is_, grid.je + 1, k]
                q2[grid.is_ - 1, grid.je + 2, k] = q2[grid.is_ + 1, grid.je + 1, k]
            if grid.ne_corner:
                q1[grid.ie + 1, grid.je + 1, k] = q1[grid.ie, grid.je + 1, k]
                q1[grid.ie + 1, grid.je + 2, k] = q1[grid.ie - 1, grid.je + 1, k]
                q2[grid.ie + 1, grid.je + 1, k] = q2[grid.ie, grid.je + 1, k]
                q2[grid.ie + 1, grid.je + 2, k] = q2[grid.ie - 1, grid.je + 1, k]
    else:
        raise ValueError("Direction not recognized. Specify either x or y")


def copy_sw_corner(q, direction, grid, kslice):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice] = q[j, grid.is_ - i + 2, kslice]
            if direction == "y":
                q[i, j, kslice] = q[grid.js - j + 2, i, kslice]


def copy_se_corner(q, direction, grid, kslice):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice] = q[grid.je + 1 - j + 2, i - grid.ie + 2, kslice]
            if direction == "y":
                q[i, j, kslice] = q[grid.je + j - 2, grid.ie + 1 - i + 2, kslice]


def copy_ne_corner(q, direction, grid, kslice):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice] = q[j, 2 * (grid.ie + 1) - 1 - i, kslice]
            if direction == "y":
                q[i, j, kslice] = q[2 * (grid.je + 1) - 1 - j, i, kslice]


def copy_nw_corner(q, direction, grid, kslice):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice] = q[grid.je + 1 - j + 2, i - 2 + grid.ie, kslice]
            if direction == "y":
                q[i, j, kslice] = q[j + 2 - grid.ie, grid.je + 1 - i + 2, kslice]


# can't actually be a stencil because offsets are variable
def copy_corners(q, direction, grid, kslice=slice(0, None)):
    if grid.sw_corner:
        copy_sw_corner(q, direction, grid, kslice)
    if grid.se_corner:
        copy_se_corner(q, direction, grid, kslice)
    if grid.ne_corner:
        copy_ne_corner(q, direction, grid, kslice)
    if grid.nw_corner:
        copy_nw_corner(q, direction, grid, kslice)


# TODO these can definitely be consolidated/made simpler
def fill_sw_corner_2d_bgrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.js - j, :] = q[grid.is_ - j, grid.js + i, :]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, :] = q[grid.is_ + i, grid.js - j, :]


def fill_nw_corner_2d_bgrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.je + 1 + j, :] = q[grid.is_ - j, grid.je + 1 - i, :]
    if direction == "y":
        q[grid.is_ - j, grid.je + 1 + i, :] = q[grid.is_ + i, grid.je + 1 + j, :]


def fill_se_corner_2d_bgrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.ie + 1 + i, grid.js - j, :] = q[grid.ie + 1 + j, grid.js + i, :]
    if direction == "y":
        q[grid.ie + 1 + j, grid.js - i, :] = q[grid.ie + 1 - i, grid.js - j, :]


def fill_ne_corner_2d_bgrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.ie + 1 + i, grid.je + 1 + j, :] = q[grid.ie + 1 + j, grid.je + 1 - i, :]
    if direction == "y":
        q[grid.ie + 1 + j, grid.je + 1 + i, :] = q[grid.ie + 1 - i, grid.je + 1 + j, :]


def fill_sw_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.js - j, :] = q[grid.is_ - j, i, :]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, :] = q[i, grid.js - j, :]


def fill_nw_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.je + j, :] = q[grid.is_ - j, grid.je - i + 1, :]
    if direction == "y":
        q[grid.is_ - j, grid.je + i, :] = q[i, grid.je + j, :]


def fill_se_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.ie + i, grid.js - j, :] = q[grid.ie + j, i, :]
    if direction == "y":
        q[grid.ie + j, grid.js - i, :] = q[grid.ie - i + 1, grid.js - j, :]


def fill_ne_corner_2d_agrid(q, i, j, direction, grid, mysign=1.0):
    if direction == "x":
        q[grid.ie + i, grid.je + j, :] = q[grid.ie + j, grid.je - i + 1, :]
    if direction == "y":
        q[grid.ie + j, grid.je + i, :] = q[grid.ie - i + 1, grid.je + j, :]


def fill_corners_2d(q, grid, gridtype, direction="x"):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if gridtype == "B":
                if grid.sw_corner:
                    fill_sw_corner_2d_bgrid(q, i, j, direction, grid)
                if grid.nw_corner:
                    fill_nw_corner_2d_bgrid(q, i, j, direction, grid)
                if grid.se_corner:
                    fill_se_corner_2d_bgrid(q, i, j, direction, grid)
                if grid.ne_corner:
                    fill_ne_corner_2d_bgrid(q, i, j, direction, grid)
            if gridtype == "A":
                if grid.sw_corner:
                    fill_sw_corner_2d_agrid(q, i, j, direction, grid)
                if grid.nw_corner:
                    fill_nw_corner_2d_agrid(q, i, j, direction, grid)
                if grid.se_corner:
                    fill_se_corner_2d_agrid(q, i, j, direction, grid)
                if grid.ne_corner:
                    fill_ne_corner_2d_agrid(q, i, j, direction, grid)


def fill_sw_corner_vector_dgrid(x, y, i, j, grid, mysign):
    x[grid.is_ - i, grid.js - j, :] = mysign * y[grid.is_ - j, i + 2, :]
    y[grid.is_ - i, grid.js - j, :] = mysign * x[j + 2, grid.js - i, :]


def fill_nw_corner_vector_dgrid(x, y, i, j, grid):
    x[grid.is_ - i, grid.je + 1 + j, :] = y[grid.is_ - j, grid.je + 1 - i, :]
    y[grid.is_ - i, grid.je + j, :] = x[j + 2, grid.je + 1 + i, :]


def fill_se_corner_vector_dgrid(x, y, i, j, grid):
    x[grid.ie + i, grid.js - j, :] = y[grid.ie + 1 + j, i + 2, :]
    y[grid.ie + 1 + i, grid.js - j, :] = x[grid.ie - j + 1, grid.js - i, :]


def fill_ne_corner_vector_dgrid(x, y, i, j, grid, mysign):
    x[grid.ie + i, grid.je + 1 + j, :] = mysign * y[grid.ie + 1 + j, grid.je - i + 1, :]
    y[grid.ie + 1 + i, grid.je + j, :] = mysign * x[grid.ie - j + 1, grid.je + 1 + i, :]


def fill_corners_dgrid(x, y, grid, vector):
    mysign = 1.0
    if vector:
        mysign = -1.0
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_vector_dgrid(x, y, i, j, grid, mysign)
            if grid.nw_corner:
                fill_nw_corner_vector_dgrid(x, y, i, j, grid)
            if grid.se_corner:
                fill_se_corner_vector_dgrid(x, y, i, j, grid)
            if grid.ne_corner:
                fill_ne_corner_vector_dgrid(x, y, i, j, grid, mysign)
