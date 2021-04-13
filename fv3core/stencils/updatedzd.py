import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FixedOriginStencil
from fv3core.stencils import basic_operations, d_sw, delnflux
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


DZ_MIN = constants.DZ_MIN


@gtscript.function
def apply_height_flux(
    geopotential_height: FloatField,
    area: FloatFieldIJ,
    x_height_flux: FloatField,
    y_height_flux: FloatField,
    x_area_flux: FloatField,
    y_area_flux: FloatField,
):
    """
    Apply the computed fluxes of height and gridcell area to the height profile.

    A positive flux of area corresponds to convergence, which expands
    the layer thickness.

    Args:
        geopotential_height: initial geopotential height
        area: gridcell area in m^2
        x_height_flux: area-weighted flux of geopotential height in x-direction,
            in units of g * m^3
        y_height_flux: area-weighted flux of geopotential height in y-direction,
            in units of g * m^3
        x_area_flux: flux of area in x-direction, in units of m^2
        y_area_flux: flux of area in y-direction, in units of m^2
    """
    area_after_flux = (
        (area + x_area_flux - x_area_flux[1, 0, 0])
        + (area + y_area_flux - y_area_flux[0, 1, 0])
        - area
    )
    # final height is the original volume plus the fluxed volumes,
    # divided by the final area
    return (
        geopotential_height * area
        + x_height_flux
        - x_height_flux[1, 0, 0]
        + y_height_flux
        - y_height_flux[0, 1, 0]
    ) / area_after_flux


def apply_geopotential_height_fluxes(
    area: FloatFieldIJ,
    initial_gz: FloatField,
    fx: FloatField,
    fy: FloatField,
    xfx_interface: FloatField,
    yfx_interface: FloatField,
    gz_x_diffusive_flux: FloatField,
    gz_y_diffusive_flux: FloatField,
    final_gz: FloatField,
    zs: FloatFieldIJ,
    ws: FloatFieldIJ,
    dt: float,
):
    """
    Apply all computed fluxes to profile of geopotential height.

    All vertically-resolved arguments are defined on the same grid
    (normally interface levels).

    Args:
        initial_gz: geopotential height profile on which to apply fluxes (in)
        fx: area-weighted flux of geopotential height in x-direction,
            in units of g * m^3
        fy: area-weighted flux of geopotential height in y-direction,
            in units of g * m^3
        xfx_interface: flux of area in x-direction, in units of m^2 (in)
        yfx_interface: flux of area in y-direction, in units of m^2 (in)
        gz_x_diffusive_flux: diffusive flux of area-weighted geopotential height
            in x-direction (in)
        gz_y_diffusive_flux: diffusive flux of area-weighted geopotential height
            in y-direction (in)
        final_gz: geopotential height (out)
        zs: surface geopotential height (in)
        ws: vertical velocity of the lowest level (to keep it at the surface) (out)
        dt: acoustic timestep (seconds) (in)
    Grid variable inputs:
        area
    """
    with computation(PARALLEL), interval(...):
        final_gz = (
            apply_height_flux(initial_gz, area, fx, fy, xfx_interface, yfx_interface)
            + (
                gz_x_diffusive_flux
                - gz_x_diffusive_flux[1, 0, 0]
                + gz_y_diffusive_flux
                - gz_y_diffusive_flux[0, 1, 0]
            )
            / area
        )

    with computation(BACKWARD):
        with interval(-1, None):
            ws = (zs - final_gz) / dt
        with interval(0, -1):
            # ensure layer thickness exceeds minimum
            other = final_gz[0, 0, 1] + DZ_MIN
            final_gz = final_gz if final_gz > other else other


def cubic_spline_interpolation_constants(
    dp0: FloatFieldK,
    gk: FloatField,
    beta: FloatField,
    gamma: FloatField,
):
    """
    Computes constants used in cubic spline interpolation
    from cell center to interface levels.

    Args:
        dp0: target pressure on interface levels (in)
        gk: interpolation constant on mid levels (out)
        beta: interpolation constant on mid levels (out)
        gamma: interpolation constant on mid levels (out)
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
        self._zh_tmp = utils.make_storage_from_shape(
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
        self._cubic_spline_interpolation_constants = FixedOriginStencil(
            cubic_spline_interpolation_constants,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self._cubic_spline_constants_initialized = False
        self._interpolate_to_layer_interface = FixedOriginStencil(
            cubic_spline_interpolation_from_layer_center_to_interfaces,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )
        self._apply_geopotential_height_fluxes = FixedOriginStencil(
            apply_geopotential_height_fluxes,
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
            dp0: air pressure on interface levels, reference pressure
                can be used as an approximation
            zs: geopotential height of surface
            zh: geopotential height defined on layer interfaces
            crx: Courant number in x-direction
            cry: Courant number in y-direction
            xfx: Area flux in x-direction
            yfx: Area flux in y-direction
            wsd: lowest layer vertical velocity required to keep layer at surface
            dt: ???
        """
        # TODO: move this logic to init, adding dp0 as an initialization argument
        if not self._cubic_spline_constants_initialized:
            self._cubic_spline_interpolation_constants(
                dp0, self._gk, self._beta, self._gamma
            )
            self._cubic_spline_constants_initialized = True
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
            self._zh_tmp,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )  # this temporary can be replaced with zh if we have selective validation
        self.finite_volume_transport(
            self._zh_tmp,
            self._crx_interface,
            self._cry_interface,
            self._xfx_interface,
            self._yfx_interface,
            self._fx,
            self._fy,
        )
        for kstart, nk in self._k_bounds:
            delnflux.compute_no_sg(
                self._zh_tmp,
                self._fx2,
                self._fy2,
                int(self._column_namelist["nord_v"][kstart]),
                self._column_namelist["damp_vt"][kstart],
                self._wk,
                kstart=kstart,
                nk=nk,
            )
        self._apply_geopotential_height_fluxes(
            self.grid.area,
            zh,
            self._fx,
            self._fy,
            self._xfx_interface,
            self._yfx_interface,
            self._fx2,
            self._fy2,
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
