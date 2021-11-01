import dace
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
from fv3core.decorators import computepath_function, computepath_method
from fv3core.utils.typing import FloatField


class CopyCorners:
    """
    Helper-class to copy corners corresponding to the fortran functions
    copy_corners_x or copy_corners_y respectively
    """

    def __init__(
        self, direction: str, temporary_field=None, origin=None, domain=None
    ) -> None:
        self.grid = spec.grid
        if origin is None:
            origin = self.grid.full_origin()

        if domain is None:
            domain = self.grid.domain_shape_full(add=(0, 0, 1))
        self._origin = origin
        self._domain = domain
        self.direction = direction
        self.kmax2 = np.empty((1,), dtype=np.int32)

    @computepath_method
    def __call__(self, field):
        """
        Fills cell quantity field using corners from itself and multipliers
        in the dirction specified initialization of the instance of this class.
        """
        copy_corners(
            field,
            self.direction,
            self.grid,
            # slice(self._origin[2] + self._domain[2])
            # slice(self._origin[2], None)
            self._origin[2],
        )


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
        self._origin = origin
        self._domain = domain
        self._direction = direction

    @computepath_method
    def __call__(self, field):
        # self.fill_corners.__call__(
        fill_corners(
            field,
            self.grid,
            "B",
            self._direction,
            slice(self._origin[2], self._origin[2] + self._domain[2]),
        )


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
            (ut[0, 0, 0] + ut[0, -1, 0])
            * ((io1 + 1) * u[0, 0, 0] - (io1 * u[-1, 0, 0]))
            + (vt[0, 0, 0] + vt[-1, 0, 0])
            * ((jo1 + 1) * v[0, 0, 0] - (jo1 * v[0, -1, 0]))
            + (
                ((jo1 + 1) * ut[0, 0, 0] - (jo1 * ut[0, -1, 0]))
                + vsign * ((io1 + 1) * vt[0, 0, 0] - (io1 * vt[-1, 0, 0]))
            )
            * ((io2 + 1) * u[0, 0, 0] - (io2 * u[-1, 0, 0]))
        )


# FOR DACE
@computepath_function
def fill_4corners(q, direction: dace.constant, grid: dace.constant):
    if direction == "x":
        for k in dace.map[0 : q.shape[2]]:
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
        for k in dace.map[0 : q.shape[2]]:
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


@computepath_function
def fill2_4corners(q1, q2, direction: dace.constant, grid: dace.constant):
    if direction == "x":
        for k in dace.map[0 : q1.shape[2]]:
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
        for k in dace.map[0 : q1.shape[2]]:
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


@computepath_function
def copy_sw_corner(
    q, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice:] = q[j, grid.is_ - i + 2, kslice:]
            if direction == "y":
                q[i, j, kslice:] = q[grid.js - j + 2, i, kslice:]


@computepath_function
def copy_se_corner(
    q, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice:] = q[grid.je + 1 - j + 2, i - grid.ie + 2, kslice:]
            if direction == "y":
                q[i, j, kslice:] = q[grid.je + j - 2, grid.ie + 1 - i + 2, kslice:]


@computepath_function
def copy_ne_corner(
    q, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice:] = q[j, 2 * (grid.ie + 1) - 1 - i, kslice:]
            if direction == "y":
                q[i, j, kslice:] = q[2 * (grid.je + 1) - 1 - j, i, kslice:]


@computepath_function
def copy_nw_corner(
    q, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice:] = q[grid.je + 1 - j + 2, i - 2 + grid.ie, kslice:]
            if direction == "y":
                q[i, j, kslice:] = q[j + 2 - grid.ie, grid.je + 1 - i + 2, kslice:]


# can't actually be a stencil because offsets are variable
@computepath_function
def copy_corners(
    q, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if grid.sw_corner:
        copy_sw_corner(q, direction, grid, kslice)
    if grid.se_corner:
        copy_se_corner(q, direction, grid, kslice)
    if grid.ne_corner:
        copy_ne_corner(q, direction, grid, kslice)
    if grid.nw_corner:
        copy_nw_corner(q, direction, grid, kslice)


# TODO these can definitely be consolidated/made simpler


@computepath_function
def fill_sw_corner_bgrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.is_ - i, grid.js - j, kslice] = q[grid.is_ - j, grid.js + i, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, kslice] = q[grid.is_ + i, grid.js - j, kslice]


@computepath_function
def fill_nw_corner_bgrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.is_ - i, grid.je + 1 + j, kslice] = q[
            grid.is_ - j, grid.je + 1 - i, kslice
        ]
    if direction == "y":
        q[grid.is_ - j, grid.je + 1 + i, kslice] = q[
            grid.is_ + i, grid.je + 1 + j, kslice
        ]


@computepath_function
def fill_se_corner_bgrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.ie + 1 + i, grid.js - j, kslice] = q[
            grid.ie + 1 + j, grid.js + i, kslice
        ]
    if direction == "y":
        q[grid.ie + 1 + j, grid.js - i, kslice] = q[
            grid.ie + 1 - i, grid.js - j, kslice
        ]


@computepath_function
def fill_ne_corner_bgrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.ie + 1 + i, grid.je + 1 + j, kslice] = q[
            grid.ie + 1 + j, grid.je + 1 - i, kslice
        ]
    if direction == "y":
        q[grid.ie + 1 + j, grid.je + 1 + i, kslice] = q[
            grid.ie + 1 - i, grid.je + 1 + j, kslice
        ]


@computepath_function
def fill_sw_corner_agrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.is_ - i, grid.js - j, kslice] = q[grid.is_ - j, i, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, kslice] = q[i, grid.js - j, kslice]


@computepath_function
def fill_nw_corner_agrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.is_ - i, grid.je + j, kslice] = q[grid.is_ - j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.je + i, kslice] = q[i, grid.je + j, kslice]


@computepath_function
def fill_se_corner_agrid(
    q, i, j, direction: dace.constant, grid: dace.constant, kslice: dace.constant
):
    if direction == "x":
        q[grid.ie + i, grid.js - j, kslice] = q[grid.ie + j, i, kslice]
    if direction == "y":
        q[grid.ie + j, grid.js - i, kslice] = q[grid.ie - i + 1, grid.js - j, kslice]


@computepath_function
def fill_ne_corner_agrid(
    q,
    i,
    j,
    direction: dace.constant,
    grid: dace.constant,
    kslice: dace.constant,
    mysign: dace.constant = 1.0,
):
    if direction == "x":
        q[grid.ie + i, grid.je + j, kslice] = q[grid.ie + j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.ie + j, grid.je + i, kslice] = q[grid.ie - i + 1, grid.je + j, kslice]


@computepath_function
def fill_corners(
    q,
    grid: dace.constant,
    gridtype: dace.constant,
    direction: dace.constant,
    kslice: dace.constant,
):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if gridtype == "B":
                if grid.sw_corner:
                    fill_sw_corner_bgrid(q, i, j, direction, grid, kslice)
                if grid.nw_corner:
                    fill_nw_corner_bgrid(q, i, j, direction, grid, kslice)
                if grid.se_corner:
                    fill_se_corner_bgrid(q, i, j, direction, grid, kslice)
                if grid.ne_corner:
                    fill_ne_corner_bgrid(q, i, j, direction, grid, kslice)
            if gridtype == "A":
                if grid.sw_corner:
                    fill_sw_corner_agrid(q, i, j, direction, grid, kslice)
                if grid.nw_corner:
                    fill_nw_corner_agrid(q, i, j, direction, grid, kslice)
                if grid.se_corner:
                    fill_se_corner_agrid(q, i, j, direction, grid, kslice)
                if grid.ne_corner:
                    fill_ne_corner_agrid(q, i, j, direction, grid, kslice)


class FillCorners:
    def __init__(self, grid, B, direction):
        self.grid = grid
        self.gridtype = B
        self.direction = direction

    @computepath_method
    def __call__(self, q, direction: dace.constant, kslice: dace.constant):
        for i in range(1, 1 + self.grid.halo):
            for j in range(1, 1 + self.grid.halo):
                if self.gridtype == "B":
                    if self.grid.sw_corner:
                        fill_sw_corner_bgrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.nw_corner:
                        fill_nw_corner_bgrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.se_corner:
                        fill_se_corner_bgrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.ne_corner:
                        fill_ne_corner_bgrid(q, i, j, direction, self.grid, kslice)
                if self.gridtype == "A":
                    if self.grid.sw_corner:
                        fill_sw_corner_agrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.nw_corner:
                        fill_nw_corner_agrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.se_corner:
                        fill_se_corner_agrid(q, i, j, direction, self.grid, kslice)
                    if self.grid.ne_corner:
                        fill_ne_corner_agrid(q, i, j, direction, self.grid, kslice)


@computepath_function
def fill_sw_corner_vector_dgrid(
    x, y, i, j, grid: dace.constant, mysign, kslice: dace.constant
):
    x[grid.is_ - i, grid.js - j, kslice] = y[grid.is_ - j, i + 2, kslice]
    x[grid.is_ - i, grid.js - j, kslice] *= mysign
    y[grid.is_ - i, grid.js - j, kslice] = x[j + 2, grid.js - i, kslice]
    y[grid.is_ - i, grid.js - j, kslice] *= mysign


@computepath_function
def fill_nw_corner_vector_dgrid(x, y, i, j, grid: dace.constant, kslice: dace.constant):
    x[grid.is_ - i, grid.je + 1 + j, kslice] = y[grid.is_ - j, grid.je + 1 - i, kslice]
    y[grid.is_ - i, grid.je + j, kslice] = x[j + 2, grid.je + 1 + i, kslice]


@computepath_function
def fill_se_corner_vector_dgrid(x, y, i, j, grid: dace.constant, kslice: dace.constant):
    x[grid.ie + i, grid.js - j, kslice] = y[grid.ie + 1 + j, i + 2, kslice]
    y[grid.ie + 1 + i, grid.js - j, kslice] = x[grid.ie - j + 1, grid.js - i, kslice]


@computepath_function
def fill_ne_corner_vector_dgrid(
    x, y, i, j, grid: dace.constant, mysign, kslice: dace.constant
):
    x[grid.ie + i, grid.je + 1 + j, kslice] = y[
        grid.ie + 1 + j, grid.je - i + 1, kslice
    ]
    x[grid.ie + i, grid.je + 1 + j, kslice] *= mysign
    y[grid.ie + 1 + i, grid.je + j, kslice] = x[
        grid.ie - j + 1, grid.je + 1 + i, kslice
    ]
    y[grid.ie + 1 + i, grid.je + j, kslice] *= mysign


@computepath_function
def fill_corners_dgrid(
    x, y, grid: dace.constant, vector, kslice: dace.constant = slice(0, None)
):
    mysign = 1.0
    if vector:
        mysign = -1.0
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_vector_dgrid(x, y, i, j, grid, mysign, kslice)
            if grid.nw_corner:
                fill_nw_corner_vector_dgrid(x, y, i, j, grid, kslice)
            if grid.se_corner:
                fill_se_corner_vector_dgrid(x, y, i, j, grid, kslice)
            if grid.ne_corner:
                fill_ne_corner_vector_dgrid(x, y, i, j, grid, mysign, kslice)
