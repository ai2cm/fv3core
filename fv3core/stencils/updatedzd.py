import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.fxadv
import fv3core.utils
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FixedOriginStencil
from fv3core.stencils import basic_operations
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


DZ_MIN = constants.DZ_MIN


@gtscript.function
def zh_base(
    z2: FloatField,
    area: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    xfx_interface: FloatField,
    yfx_interface: FloatField,
):
    area_after_flux = (
        area
        + xfx_interface
        - xfx_interface[1, 0, 0]
        + yfx_interface
        - yfx_interface[0, 1, 0]
    )
    return (z2 * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / area_after_flux


def zh_damp(
    area: FloatFieldIJ,
    z2: FloatField,
    fx: FloatField,
    fy: FloatField,
    xfx_interface: FloatField,
    yfx_interface: FloatField,
    fx2: FloatField,
    fy2: FloatField,
    rarea: FloatFieldIJ,
    zh: FloatField,
    zs: FloatFieldIJ,
    ws: FloatFieldIJ,
    dt: float,
):
    """Update geopotential height due to area average flux divergence
    Args:
        z2: zh that has been advected forward in time (in)
        fx: Flux in the x direction that transported z2 (in)
        fy: Flux in the y direction that transported z2 (in)
        xfx_interface: Area flux per timestep in x-direction (in)
        yfx_interface: Area flux per timestep in y-direction (in)
        fx2: diffusive flux in the x-direction (in)
        fy2: diffusive flux in the y-direction (in)
        zh: geopotential height (out)
        zs: surface geopotential height (in)
        ws: vertical velocity of the lowest level (to keep it at the surface) (out)
        dt: acoustic timestep (seconds) (in)
    Grid variable inputs:
        area
        rarea
    """
    with computation(PARALLEL), interval(...):
        zhbase = zh_base(z2, area, fx, fy, xfx_interface, yfx_interface)
        zh = zhbase + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
    with computation(BACKWARD):
        with interval(-1, None):
            ws = (zs - zh) * 1.0 / dt
        with interval(0, -1):
            other = zh[0, 0, 1] + DZ_MIN
            zh = zh if zh > other else other


def cubic_spline_interpolation_constants(
    dp0: FloatFieldK,
    gk: FloatField,
    beta: FloatField,
    gamma: FloatField,
):
    """
    Computes constants used in cubic spline interpolation.
    """
    with computation(FORWARD):
        with interval(0, 1):
            gk = dp0[1] / dp0
            beta = gk * (gk + 0.5)
            gamma = (1.0 + gk * (gk + 1.5)) / beta
        with interval(1, -1):
            gk = dp0[-1] / dp0
            beta = 2.0 + 2.0 * gk - gamma[0, 0, -1]
            gamma = gk / beta


def cubic_spline_interpolation_from_layer_center_to_interfaces(
    q_center: FloatField,
    q_interface: FloatField,
    gk: FloatField,
    beta: FloatField,
    gamma: FloatField,
) -> FloatField:
    """
    Interpolate a field from layer (vertical) centers to interfaces.

    Args:
        q_center (in): value on layer centers
        q_interface (out): value on layer interfaces
        gk (in): cubic spline interpolation constant
        beta (in): cubic spline interpolation constant
        gamma (in): cubic spline interpolation constant
    """
    with computation(FORWARD):
        with interval(0, 1):
            xt1 = 2.0 * gk * (gk + 1.0)
            q_interface = (xt1 * q_center + q_center[0, 0, 1]) / beta
        with interval(1, -1):
            q_interface = (
                3.0 * (q_center[0, 0, -1] + gk * q_center) - q_interface[0, 0, -1]
            ) / beta
        with interval(-1, None):
            a_bot = 1.0 + gk[0, 0, -1] * (gk[0, 0, -1] + 1.5)
            xt1 = 2.0 * gk[0, 0, -1] * (gk[0, 0, -1] + 1.0)
            xt2 = gk[0, 0, -1] * (gk[0, 0, -1] + 0.5) - a_bot * gamma[0, 0, -1]
            q_interface = (
                xt1 * q_center[0, 0, -1]
                + q_center[0, 0, -2]
                - a_bot * q_interface[0, 0, -1]
            ) / xt2
    with computation(BACKWARD), interval(0, -1):
        q_interface -= gamma * q_interface[0, 0, 1]


class UpdateDeltaZOnDGrid:
    """
    Fortran name is updatedzd.
    """

    def __init__(self, grid, column_namelist, k_bounds):
        self.grid = spec.grid
        self._column_namelist = column_namelist
        if any(
            column_namelist["damp_vt"][kstart] <= 1e-5
            for kstart in range(len(k_bounds))
        ):
            raise NotImplementedError("damp <= 1e-5 in column_cols is untested")
        self._k_bounds = k_bounds  # d_sw.k_bounds()
        largest_possible_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._crx_interface = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(0, -self.grid.halo, 0))
        )
        self._cry_interface = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(-self.grid.halo, 0, 0))
        )
        self._xfx_interface = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(0, -self.grid.halo, 0))
        )
        self._yfx_interface = utils.make_storage_from_shape(
            largest_possible_shape, grid.compute_origin(add=(-self.grid.halo, 0, 0))
        )
        self._wk = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fx2 = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fy2 = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fx = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._fy = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._zh_intermediate = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._gk = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._gamma = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )
        self._beta = utils.make_storage_from_shape(
            largest_possible_shape, grid.full_origin()
        )

        self.finite_volume_transport = FiniteVolumeTransport(
            spec.namelist, spec.namelist.hord_tm
        )
        ax_offsets = fv3core.utils.axis_offsets(
            self.grid, self.grid.full_origin(), self.grid.domain_shape_full()
        )
        self._cubic_spline_interpolation_constants = FixedOriginStencil(
            cubic_spline_interpolation_constants,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self._interpolate_to_layer_interface = FixedOriginStencil(
            cubic_spline_interpolation_from_layer_center_to_interfaces,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self._zh_damp = FixedOriginStencil(
            zh_damp,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 0, 1)),
        )

    def __call__(
        self,
        dp0: FloatFieldK,
        zs: FloatFieldIJ,
        zh: FloatField,
        crx: FloatField,
        cry: FloatField,
        xfx: FloatField,
        yfx: FloatField,
        wsd: FloatFieldIJ,
        dt: float,
    ):
        """
        Args:
            dp0: ???
            zs: ???
            zh: ???
            crx: Courant number in x-direction
            cry: Courant number in y-direction
            xfx: Mass flux in x-direction
            yfx: Mass flux in y-direction
            wsd: ???
            dt: ???
        """
        self._cubic_spline_interpolation_constants(
            dp0, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            crx, self._crx_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            xfx, self._xfx_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            cry, self._cry_interface, self._gk, self._beta, self._gamma
        )
        self._interpolate_to_layer_interface(
            yfx, self._yfx_interface, self._gk, self._beta, self._gamma
        )
        basic_operations.copy_stencil(
            zh,
            self._zh_intermediate,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self.finite_volume_transport(
            self._zh_intermediate,
            self._crx_interface,
            self._cry_interface,
            self._xfx_interface,
            self._yfx_interface,
            self._fx,
            self._fy,
        )
        for kstart, nk in self._k_bounds:
            delnflux.compute_no_sg(
                self._zh_intermediate,
                self._fx2,
                self._fy2,
                int(self._column_namelist["nord_v"][kstart]),
                self._column_namelist["damp_vt"][kstart],
                self._wk,
                kstart=kstart,
                nk=nk,
            )
        self._zh_damp(
            self.grid.area,
            self._zh_intermediate,
            self._fx,
            self._fy,
            self._xfx_interface,
            self._yfx_interface,
            self._fx2,
            self._fy2,
            self.grid.rarea,
            zh,
            zs,
            wsd,
            dt,
        )


def compute(
    dp0: FloatFieldK,
    zs: FloatFieldIJ,
    zh: FloatField,
    crx: FloatField,
    cry: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    wsd: FloatFieldIJ,
    dt: float,
):
    updatedzd = utils.cached_stencil_class(UpdateDeltaZOnDGrid)(
        spec.grid, d_sw.get_column_namelist(), d_sw.k_bounds()
    )
    updatedzd(dp0, zs, zh, crx, cry, xfx, yfx, wsd, dt)
