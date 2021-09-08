from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

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

    def __init__(self, direction: str, temporary_field=None) -> None:
        self.grid = spec.grid
        """The grid for this stencil"""

        origin = self.grid.full_origin()
        """The origin for the corner computation"""

        domain = self.grid.domain_shape_full(add=(0, 0, 1))
        """The full domain required to do corner computation everywhere"""

        if temporary_field is not None:
            self._corner_tmp = temporary_field
        else:
            self._corner_tmp = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
            )

        self._copy_full_domain = FrozenStencil(
            func=copy_defn,
            origin=origin,
            domain=domain,
        )
        """Stencil Wrapper to do the copy of the input field to the temporary field"""

        ax_offsets = axis_offsets(spec.grid, origin, domain)
        if direction == "x":
            self._copy_corners = FrozenStencil(
                func=copy_corners_x_stencil_defn,
                origin=origin,
                domain=domain,
                externals={
                    **ax_offsets,
                },
            )
        elif direction == "y":
            self._copy_corners = FrozenStencil(
                func=copy_corners_y_stencil_defn,
                origin=origin,
                domain=domain,
                externals={
                    **ax_offsets,
                },
            )
        else:
            raise ValueError("Direction must be either 'x' or 'y'")

    def __call__(self, field: FloatField):
        """
        Fills cell quantity field using corners from itself and multipliers
        in the dirction specified initialization of the instance of this class.
        """
        self._copy_full_domain(field, self._corner_tmp)
        self._copy_corners(self._corner_tmp, field)


@gtscript.function
def fill_corners_2cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[0, 1, 0]
    with horizontal(region[i_start - 2, j_start - 1]):
        q = sw_mult * q_corner[1, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[0, 1, 0]
    with horizontal(region[i_end + 2, j_start - 1]):
        q = se_mult * q_corner[-1, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[0, -1, 0]
    with horizontal(region[i_start - 2, j_end + 1]):
        q = nw_mult * q_corner[1, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[0, -1, 0]
    with horizontal(region[i_end + 2, j_end + 1]):
        q = ne_mult * q_corner[-1, -2, 0]

    return q


def fill_corners_2cells_x_stencil(q_out: FloatField, q_in: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = fill_corners_2cells_mult_x(q_out, q_in, 1.0, 1.0, 1.0, 1.0)


def fill_corners_2cells_y_stencil(q_out: FloatField, q_in: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = fill_corners_2cells_mult_y(q_out, q_in, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_2cells_x(q: FloatField):
    """
    Fills cell quantity q in x-dir.
    """
    return fill_corners_2cells_mult_x(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_x(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 3, j_start - 1]):
        q = sw_mult * q_corner[2, 3, 0]

    # Southeast
    with horizontal(region[i_end + 3, j_start - 1]):
        q = se_mult * q_corner[-2, 3, 0]

    # Northwest
    with horizontal(region[i_start - 3, j_end + 1]):
        q = nw_mult * q_corner[2, -3, 0]

    # Northeast
    with horizontal(region[i_end + 3, j_end + 1]):
        q = ne_mult * q_corner[-2, -3, 0]

    return q


@gtscript.function
def fill_corners_2cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_start - 2]):
        q = sw_mult * q_corner[2, 1, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_start - 2]):
        q = se_mult * q_corner[-2, 1, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_end + 2]):
        q = nw_mult * q_corner[2, -1, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_end + 2]):
        q = ne_mult * q_corner[-2, -1, 0]

    return q


@gtscript.function
def fill_corners_2cells_y(q: FloatField):
    """
    Fills cell quantity q in y-dir.
    """
    return fill_corners_2cells_mult_y(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_y(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 1, j_start - 3]):
        q = sw_mult * q_corner[3, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 3]):
        q = se_mult * q_corner[-3, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 3]):
        q = nw_mult * q_corner[3, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 3]):
        q = ne_mult * q_corner[-3, -2, 0]

    return q


def copy_corners_x_stencil_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q_out = q_in[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q_out = q_in[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q_out = q_in[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q_out = q_in[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q_out = q_in[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q_out = q_in[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q_out = q_in[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q_out = q_in[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q_out = q_in[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q_out = q_in[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q_out = q_in[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q_out = q_in[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q_out = q_in[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q_out = q_in[-2, -3, 0]


def copy_corners_y_stencil_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q_out = q_in[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q_out = q_in[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q_out = q_in[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q_out = q_in[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q_out = q_in[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q_out = q_in[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q_out = q_in[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q_out = q_in[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q_out = q_in[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q_out = q_in[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q_out = q_in[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q_out = q_in[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q_out = q_in[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q_out = q_in[-3, -2, 0]


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

        if temporary_field is not None:
            self._corner_tmp = temporary_field
        else:
            self._corner_tmp = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
            )

        self._copy_full_domain = FrozenStencil(
            func=copy_defn,
            origin=origin,
            domain=domain,
        )

        """Stencil Wrapper to do the copy of the input field to the temporary field"""

        ax_offsets = axis_offsets(self.grid, origin, domain)

        if direction == "x":
            self._fill_corners_bgrid = FrozenStencil(
                func=fill_corners_bgrid_x_defn,
                origin=origin,
                domain=domain,
                externals=ax_offsets,
            )
        elif direction == "y":
            self._fill_corners_bgrid = FrozenStencil(
                func=fill_corners_bgrid_y_defn,
                origin=origin,
                domain=domain,
                externals=ax_offsets,
            )

        else:
            raise ValueError("Direction must be either 'x' or 'y'")

    def __call__(self, field: FloatField):
        self._copy_full_domain(field, self._corner_tmp)
        self._fill_corners_bgrid(self._corner_tmp, field)


def fill_corners_bgrid_x_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw and se corner
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 2, j_start - 1]
        ):
            q_out = q_in[0, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 3, j_start - 1]
        ):
            q_out = q_in[-1, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 4, j_start - 1]
        ):
            q_out = q_in[-2, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 2, j_start - 2]
        ):
            q_out = q_in[1, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 3, j_start - 2]
        ):
            q_out = q_in[0, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 4, j_start - 2]
        ):
            q_out = q_in[-1, 5, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 2, j_start - 3]
        ):
            q_out = q_in[2, 4, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 3, j_start - 3]
        ):
            q_out = q_in[1, 5, 0]
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 4, j_start - 3]
        ):
            q_out = q_in[0, 6, 0]
        # nw and ne corner
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 2]):
            q_out = q_in[0, -2, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 2]):
            q_out = q_in[-1, -3, 0]
        with horizontal(region[i_start - 1, j_end + 4], region[i_end + 4, j_end + 2]):
            q_out = q_in[-2, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 3]):
            q_out = q_in[1, -3, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 3]):
            q_out = q_in[0, -4, 0]
        with horizontal(region[i_start - 2, j_end + 4], region[i_end + 4, j_end + 3]):
            q_out = q_in[-1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 4]):
            q_out = q_in[2, -4, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 4]):
            q_out = q_in[1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 4], region[i_end + 4, j_end + 4]):
            q_out = q_in[0, -6, 0]


def fill_corners_bgrid_y_defn(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw and nw corners
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q_out = q_in[2, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q_out = q_in[3, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q_out = q_in[4, 2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q_out = q_in[3, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q_out = q_in[4, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q_out = q_in[5, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 4]
        ):
            q_out = q_in[4, -2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 4]
        ):
            q_out = q_in[5, -1, 0]
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 4]
        ):
            q_out = q_in[6, 0, 0]
        # se and ne corners
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 2, j_end + 2]):
            q_out = q_in[-2, 0, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 3, j_end + 2]):
            q_out = q_in[-3, 1, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 4, j_end + 2]):
            q_out = q_in[-4, 2, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 2, j_end + 3]):
            q_out = q_in[-3, -1, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 3, j_end + 3]):
            q_out = q_in[-4, 0, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 4, j_end + 3]):
            q_out = q_in[-5, 1, 0]
        with horizontal(region[i_end + 4, j_start - 1], region[i_end + 2, j_end + 4]):
            q_out = q_in[-4, -2, 0]
        with horizontal(region[i_end + 4, j_start - 2], region[i_end + 3, j_end + 4]):
            q_out = q_in[-5, -1, 0]
        with horizontal(region[i_end + 4, j_start - 3], region[i_end + 4, j_end + 4]):
            q_out = q_in[-6, 0, 0]


# TODO these fill corner 2d, agrid, bgrid routines need to be tested and integrated;
# they've just been copied from an older version of the code

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
        q[grid.ie + 1 + i, grid.je + 1 + j :] = q[grid.ie + 1 + j, grid.je + 1 - i, :]
    if direction == "y":
        q[grid.ie + 1 + i, grid.je + 1 + j :] = q[grid.ie + 1 - i, grid.je + 1 + j, :]


def fill_sw_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.is_ - i, grid.js - j, kslice] = q[grid.is_ - j, grid.js + i - 1, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, kslice] = q[grid.is_ + i - 1, grid.js - j, kslice]


def fill_nw_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.is_ - i, grid.je + j, kslice] = q[grid.is_ - j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.is_ - j, grid.je + i, kslice] = q[grid.is_ + i - 1, grid.je + j, kslice]


def fill_se_corner_2d_agrid(q, i, j, direction, grid, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.ie + i, grid.js - j, kslice] = q[grid.ie + j, grid.is_ + i - 1, kslice]
    if direction == "y":
        q[grid.ie + j, grid.js - i, kslice] = q[grid.ie - i + 1, grid.js - j, kslice]


def fill_ne_corner_2d_agrid(q, i, j, direction, grid, mysign=1.0, kstart=0, nk=None):
    kslice, nk = utils.kslice_from_inputs(kstart, nk, grid)
    if direction == "x":
        q[grid.ie + i, grid.je + j, kslice] = q[grid.ie + j, grid.je - i + 1, kslice]
    if direction == "y":
        q[grid.ie + j, grid.je + i, kslice] = q[grid.ie - i + 1, grid.je + j, kslice]


def fill_corners_2d(q, grid, gridtype, direction="x"):
    if gridtype == "B":
        fill_corners_2d_bgrid(q, grid, gridtype, direction)
    elif gridtype == "A":
        fill_corners_2d_agrid(q, grid, gridtype, direction)
    else:
        raise NotImplementedError()

def fill_corners_2d_bgrid(q, grid, gridtype, direction="x"):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_2d_bgrid(q, i, j, direction, grid)
            if grid.nw_corner:
                fill_nw_corner_2d_bgrid(q, i, j, direction, grid)
            if grid.se_corner:
                fill_se_corner_2d_bgrid(q, i, j, direction, grid)
            if grid.ne_corner:
                fill_ne_corner_2d_bgrid(q, i, j, direction, grid)

def fill_corners_2d_agrid(q, grid, gridtype, direction="x"):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.nw_corner:
                fill_nw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.se_corner:
                fill_se_corner_2d_agrid(q, i, j, direction, grid)
            if grid.ne_corner:
                fill_ne_corner_2d_agrid(q, i, j, direction, grid)



def fill_corners_agrid(x, y, grid, vector):
    if vector:
        mysign = -1.0
    else:
        mysign = 1.0
    i_end = grid.halo + grid.npx - 2  # index of last value in compute domain
    j_end = grid.halo + grid.npy - 2
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                x[grid.halo - i, grid.halo - j, :] = (
                    mysign * y[grid.halo - j, grid.halo - 1 + i, :]
                )
                y[grid.halo - j, grid.halo - i, :] = (
                    mysign * x[grid.halo - 1 + i, grid.halo - j, :]
                )
            if grid.nw_corner:
                x[grid.halo - i, j_end + j, :] = y[grid.halo - j, j_end - i + 1, :]
                y[grid.halo - j, j_end + i, :] = x[grid.halo - 1 + i, j_end + j, :]
            if grid.se_corner:
                x[i_end + i, grid.halo - j, :] = y[i_end + j, grid.halo - 1 + i, :]
                y[i_end + j, grid.halo - i, :] = x[i_end - i + 1, grid.halo - j, :]
            if grid.ne_corner:
                x[i_end + i, j_end + j, :] = mysign * y[i_end + j, j_end - i + 1, :]
                y[i_end + j, j_end + i, :] = mysign * x[i_end - i + 1, j_end + j, :]


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


def fill_sw_corner_vector_cgrid(x, y, i, j, grid):
    x[grid.is_ - i, grid.js - j, :] = y[j + 2, grid.js - i, :]
    y[grid.is_ - i, grid.js - j, :] = x[grid.is_ - j, i + 2, :]


def fill_nw_corner_vector_cgrid(x, y, i, j, grid, mysign):
    x[grid.is_ - i, grid.je + j, :]     = mysign * y[j + 2,        grid.je + 1 + i, :]
    y[grid.is_ - i, grid.je + 1 + j, :] = mysign * x[grid.is_ - j, grid.je + 1 - i, :]


def fill_se_corner_vector_cgrid(x, y, i, j, grid, mysign):
    x[grid.ie + 1 + i, grid.js - j, :] = mysign * y[grid.ie + 1 - j, grid.js - i, :]
    y[grid.ie + i, grid.js - j, :] = mysign * x[grid.ie + 1 + j, i + 2, :]


def fill_ne_corner_vector_cgrid(x, y, i, j, grid):
    x[grid.ie + 1 + i, grid.je + j, :] = y[grid.ie + 1 - j, grid.je + 1 + i, :]
    y[grid.ie + i, grid.je + 1 + j, :] = x[grid.ie + 1 + j, grid.je + 1 - i, :]


def fill_corners_cgrid(x, y, grid, vector):
    mysign = 1.0
    if vector:
        mysign = -1.0
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_vector_cgrid(x, y, i, j, grid)
            if grid.nw_corner:
                fill_nw_corner_vector_cgrid(x, y, i, j, grid, mysign)
            if grid.se_corner:
                fill_se_corner_vector_cgrid(x, y, i, j, grid, mysign)
            if grid.ne_corner:
                fill_ne_corner_vector_cgrid(x, y, i, j, grid)


def fill_corners_dgrid_defn(
    x_in: FloatField,
    x_out: FloatField,
    y_in: FloatField,
    y_out: FloatField,
    mysign: float,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # sw corner
        with horizontal(region[i_start - 1, j_start - 1]):
            x_out = mysign * y_in[0, 1, 0]
        with horizontal(region[i_start - 1, j_start - 1]):
            y_out = mysign * x_in[1, 0, 0]
        with horizontal(region[i_start - 1, j_start - 2]):
            x_out = mysign * y_in[-1, 2, 0]
        with horizontal(region[i_start - 1, j_start - 2]):
            y_out = mysign * x_in[2, 1, 0]
        with horizontal(region[i_start - 1, j_start - 3]):
            x_out = mysign * y_in[-2, 3, 0]
        with horizontal(region[i_start - 1, j_start - 3]):
            y_out = mysign * x_in[3, 2, 0]
        with horizontal(region[i_start - 2, j_start - 1]):
            x_out = mysign * y_in[1, 2, 0]
        with horizontal(region[i_start - 2, j_start - 1]):
            y_out = mysign * x_in[2, -1, 0]
        with horizontal(region[i_start - 2, j_start - 2]):
            x_out = mysign * y_in[0, 3, 0]
        with horizontal(region[i_start - 2, j_start - 2]):
            y_out = mysign * x_in[3, 0, 0]
        with horizontal(region[i_start - 2, j_start - 3]):
            x_out = mysign * y_in[-1, 4, 0]
        with horizontal(region[i_start - 2, j_start - 3]):
            y_out = mysign * x_in[4, 1, 0]
        with horizontal(region[i_start - 3, j_start - 1]):
            x_out = mysign * y_in[2, 3, 0]
        with horizontal(region[i_start - 3, j_start - 1]):
            y_out = mysign * x_in[3, -2, 0]
        with horizontal(region[i_start - 3, j_start - 2]):
            x_out = mysign * y_in[1, 4, 0]
        with horizontal(region[i_start - 3, j_start - 2]):
            y_out = mysign * x_in[4, -1, 0]
        with horizontal(region[i_start - 3, j_start - 3]):
            x_out = mysign * y_in[0, 5, 0]
        with horizontal(region[i_start - 3, j_start - 3]):
            y_out = mysign * x_in[5, 0, 0]
        # ne corner
        with horizontal(region[i_end + 1, j_end + 2]):
            x_out = mysign * y_in[1, -2, 0]
        with horizontal(region[i_end + 2, j_end + 1]):
            y_out = mysign * x_in[-2, 1, 0]
        with horizontal(region[i_end + 1, j_end + 3]):
            x_out = mysign * y_in[2, -3, 0]
        with horizontal(region[i_end + 2, j_end + 2]):
            y_out = mysign * x_in[-3, 0, 0]
        with horizontal(region[i_end + 1, j_end + 4]):
            x_out = mysign * y_in[3, -4, 0]
        with horizontal(region[i_end + 2, j_end + 3]):
            y_out = mysign * x_in[-4, -1, 0]
        with horizontal(region[i_end + 2, j_end + 2]):
            x_out = mysign * y_in[0, -3, 0]
        with horizontal(region[i_end + 3, j_end + 1]):
            y_out = mysign * x_in[-3, 2, 0]
        with horizontal(region[i_end + 2, j_end + 3]):
            x_out = mysign * y_in[1, -4, 0]
        with horizontal(region[i_end + 3, j_end + 2]):
            y_out = mysign * x_in[-4, 1, 0]
        with horizontal(region[i_end + 2, j_end + 4]):
            x_out = mysign * y_in[2, -5, 0]
        with horizontal(region[i_end + 3, j_end + 3]):
            y_out = mysign * x_in[-5, 0, 0]
        with horizontal(region[i_end + 3, j_end + 2]):
            x_out = mysign * y_in[-1, -4, 0]
        with horizontal(region[i_end + 4, j_end + 1]):
            y_out = mysign * x_in[-4, 3, 0]
        with horizontal(region[i_end + 3, j_end + 3]):
            x_out = mysign * y_in[0, -5, 0]
        with horizontal(region[i_end + 4, j_end + 2]):
            y_out = mysign * x_in[-5, 2, 0]
        with horizontal(region[i_end + 3, j_end + 4]):
            x_out = mysign * y_in[1, -6, 0]
        with horizontal(region[i_end + 4, j_end + 3]):
            y_out = mysign * x_in[-6, 1, 0]
        # nw corner
        with horizontal(region[i_start - 1, j_end + 2]):
            x_out = y_in[0, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1]):
            y_out = x_in[1, 1, 0]
        with horizontal(region[i_start - 1, j_end + 3]):
            x_out = y_in[-1, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2]):
            y_out = x_in[2, 0, 0]
        with horizontal(region[i_start - 1, j_end + 4]):
            x_out = y_in[-2, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3]):
            y_out = x_in[3, -1, 0]
        with horizontal(region[i_start - 2, j_end + 2]):
            x_out = y_in[1, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1]):
            y_out = x_in[2, 2, 0]
        with horizontal(region[i_start - 2, j_end + 3]):
            x_out = y_in[0, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2]):
            y_out = x_in[3, 1, 0]
        with horizontal(region[i_start - 2, j_end + 4]):
            x_out = y_in[-1, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3]):
            y_out = x_in[4, 0, 0]
        with horizontal(region[i_start - 3, j_end + 2]):
            x_out = y_in[2, -4, 0]
        with horizontal(region[i_start - 3, j_end + 1]):
            y_out = x_in[3, 3, 0]
        with horizontal(region[i_start - 3, j_end + 3]):
            x_out = y_in[1, -5, 0]
        with horizontal(region[i_start - 3, j_end + 2]):
            y_out = x_in[4, 2, 0]
        with horizontal(region[i_start - 3, j_end + 4]):
            x_out = y_in[0, -6, 0]
        with horizontal(region[i_start - 3, j_end + 3]):
            y_out = x_in[5, 1, 0]
        # se corner
        with horizontal(region[i_end + 1, j_start - 1]):
            x_out = y_in[1, 1, 0]
        with horizontal(region[i_end + 2, j_start - 1]):
            y_out = x_in[-2, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2]):
            x_out = y_in[2, 2, 0]
        with horizontal(region[i_end + 2, j_start - 2]):
            y_out = x_in[-3, 1, 0]
        with horizontal(region[i_end + 1, j_start - 3]):
            x_out = y_in[3, 3, 0]
        with horizontal(region[i_end + 2, j_start - 3]):
            y_out = x_in[-4, 2, 0]
        with horizontal(region[i_end + 2, j_start - 1]):
            x_out = y_in[0, 2, 0]
        with horizontal(region[i_end + 3, j_start - 1]):
            y_out = x_in[-3, -1, 0]
        with horizontal(region[i_end + 2, j_start - 2]):
            x_out = y_in[1, 3, 0]
        with horizontal(region[i_end + 3, j_start - 2]):
            y_out = x_in[-4, 0, 0]
        with horizontal(region[i_end + 2, j_start - 3]):
            x_out = y_in[2, 4, 0]
        with horizontal(region[i_end + 3, j_start - 3]):
            y_out = x_in[-5, 1, 0]
        with horizontal(region[i_end + 3, j_start - 1]):
            x_out = y_in[-1, 3, 0]
        with horizontal(region[i_end + 4, j_start - 1]):
            y_out = x_in[-4, -2, 0]
        with horizontal(region[i_end + 3, j_start - 2]):
            x_out = y_in[0, 4, 0]
        with horizontal(region[i_end + 4, j_start - 2]):
            y_out = x_in[-5, -1, 0]
        with horizontal(region[i_end + 3, j_start - 3]):
            x_out = y_in[1, 5, 0]
        with horizontal(region[i_end + 4, j_start - 3]):
            y_out = x_in[-6, 0, 0]


@gtscript.function
def corner_ke(
    u,
    v,
    ut,
    vt,
    dt,
    io1,
    jo1,
    io2,
    vsign,
):
    dt6 = dt / 6.0

    return dt6 * (
        (ut[0, 0, 0] + ut[0, -1, 0]) * ((io1 + 1) * u[0, 0, 0] - (io1 * u[-1, 0, 0]))
        + (vt[0, 0, 0] + vt[-1, 0, 0]) * ((jo1 + 1) * v[0, 0, 0] - (jo1 * v[0, -1, 0]))
        + (
            ((jo1 + 1) * ut[0, 0, 0] - (jo1 * ut[0, -1, 0]))
            + vsign * ((io1 + 1) * vt[0, 0, 0] - (io1 * vt[-1, 0, 0]))
        )
        * ((io2 + 1) * u[0, 0, 0] - (io2 * u[-1, 0, 0]))
    )
