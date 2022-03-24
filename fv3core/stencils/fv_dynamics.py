from types import SimpleNamespace
from typing import Mapping

# [DaCe] dace.constant
from dace import constant as dace_constant
from dace.frontend.python.interface import nounroll as dace_no_unroll
from fv3core.utils.dace.computepath import computepath_method, dace_inhibitor
from fv3core.utils.dace.utils import (
    cb_nvtx_range_pop,
    cb_nvtx_range_push_dynsteps,
)
import fv3core._config as spec

from gt4py.gtscript import PARALLEL, computation, interval, log

import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
from fv3core._config import DynamicalCoreConfig
from fv3core.decorators import ArgSpec, FrozenStencil, get_namespace
from fv3core.stencils import fvtp2d, tracer_2d_1l
from fv3core.stencils.basic_operations import copy_defn
from fv3core.stencils.c2l_ord import CubedToLatLon
from fv3core.stencils.del2cubed import HyperdiffusionDamping
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from fv3core.stencils.remapping import LagrangianToEulerian
from fv3core.utils import global_config
from fv3core.utils.grid import DampingCoefficients, GridData
from fv3core.utils.stencil import StencilFactory
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
from fv3gfs.util.halo_updater import HaloUpdater

# nq is actually given by ncnst - pnats, where those are given in atmosphere.F90 by:
# ncnst = Atm(mytile)%ncnst
# pnats = Atm(mytile)%flagstruct%pnats
# here we hard-coded it because 8 is the only supported value, refactor this later!
NQ = 8  # state.nq_tot - spec.namelist.dnats


def pt_adjust(pkz: FloatField, dp1: FloatField, q_con: FloatField, pt: FloatField):
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz


def set_omega(delp: FloatField, delz: FloatField, w: FloatField, omga: FloatField):
    with computation(PARALLEL), interval(...):
        omga = delp / delz * w


def init_pfull(
    ak: FloatFieldK,
    bk: FloatFieldK,
    pfull: FloatField,
    p_ref: float,
):
    with computation(PARALLEL), interval(...):
        ph1 = ak + bk * p_ref
        ph2 = ak[1] + bk[1] * p_ref
        pfull = (ph2 - ph1) / log(ph2 / ph1)


def fvdyn_temporaries(quantity_factory: fv3gfs.util.QuantityFactory, shape, grid):
    tmps = {}
    for name in ["te_2d", "te0_2d", "wsd"]:
        quantity = quantity_factory.empty(
            dims=[fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM], units="unknown"
        )
        tmps[f"{name}_quantity"] = quantity
        tmps[name] = quantity.storage
    for name in ["cappa", "dp1", "cvm"]:
        quantity = quantity_factory.empty(
            dims=[fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM],
            units="unknown",
        )
        tmps[f"{name}_quantity"] = quantity
        tmps[name] = quantity.storage
    gz = quantity_factory.empty(dims=[fv3gfs.util.Z_DIM], units="m^2 s^-2")
    tmps["gz_quantity"] = gz
    tmps["gz"] = gz.storage
    return tmps


# [DaCe] Split the argspec out to reference it without needing to reference
# dycore itself
class DynamicalCoreArgSpec:
    values = (
        ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
        ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
        ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
        ArgSpec("pt", "air_temperature", "degK", intent="inout"),
        ArgSpec(
            "delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"
        ),
        ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
        ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
        ArgSpec("u", "x_wind", "m/s", intent="inout"),
        ArgSpec("v", "y_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("uc", "x_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("vc", "y_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("q_con", "total_condensate_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
        ArgSpec("phis", "surface_geopotential", "m^2 s^-2", intent="in"),
        ArgSpec(
            "pk",
            "interface_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec(
            "pkz",
            "layer_mean_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec("ps", "surface_pressure", "Pa", intent="inout"),
        ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
        ArgSpec("mfxd", "accumulated_x_mass_flux", "unknown", intent="inout"),
        ArgSpec("mfyd", "accumulated_y_mass_flux", "unknown", intent="inout"),
        ArgSpec("cxd", "accumulated_x_courant_number", "", intent="inout"),
        ArgSpec("cyd", "accumulated_y_courant_number", "", intent="inout"),
        ArgSpec(
            "diss_estd",
            "dissipation_estimate_from_heat_source",
            "unknown",
            intent="inout",
        ),
    )


class DynamicalCore:
    """
    Corresponds to fv_dynamics in original Fortran sources.
    """

    def __init__(
        self,
        comm: fv3gfs.util.CubedSphereCommunicator,
        grid_data: GridData,
        stencil_factory: StencilFactory,
        damping_coefficients: DampingCoefficients,
        config: DynamicalCoreConfig,
        ak: FloatField,
        bk: FloatField,
        phis: fv3gfs.util.Quantity,
        state: SimpleNamespace,
        timer: fv3gfs.util.Timer = fv3gfs.util.NullTimer(),
    ):
        """
        Args:
            comm: object for cubed sphere inter-process communication
            grid_data: metric terms defining the model grid
            stencil_factory: creates stencils
            damping_coefficients: damping configuration/constants
            config: configuration of dynamical core, for example as would be set by
                the namelist in the Fortran model
            ak: atmosphere hybrid a coordinate (Pa)
            bk: atmosphere hybrid b coordinate (dimensionless)
            phis: surface geopotential height
        """
        # nested and stretched_grid are options in the Fortran code which we
        # have not implemented, so they are hard-coded here.
        nested = False
        stretched_grid = False
        grid_indexing = stencil_factory.grid_indexing
        sizer = fv3gfs.util.SubtileGridSizer.from_tile_params(
            nx_tile=config.npx - 1,
            ny_tile=config.npy - 1,
            nz=config.npz,
            n_halo=grid_indexing.n_halo,
            layout=config.layout,
            tile_partitioner=comm.tile.partitioner,
            tile_rank=comm.tile.rank,
            extra_dim_lengths={},
        )
        quantity_factory = fv3gfs.util.QuantityFactory.from_backend(
            sizer, backend=global_config.get_backend()
        )
        assert config.moist_phys, "fvsetup is only implemented for moist_phys=true"
        assert config.nwat == 6, "Only nwat=6 has been implemented and tested"
        # [DaCe] self.comm not useful, dace is trying to deep_copy the entire comm
        #        when only the rank is used
        self.comm_rank = comm.rank
        self.grid_data = grid_data
        self.grid_indexing = grid_indexing
        self._da_min = damping_coefficients.da_min
        self.config = config

        tracer_transport = fvtp2d.FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_tr,
        )

        # [DaCe] Build tracers names & storages
        self.tracers = {}
        for name in utils.tracer_variables[0:NQ]:
            self.tracers[name] = state.__dict__[name + "_quantity"]
        self.tracer_storages = {
            name: quantity.storage for name, quantity in self.tracers.items()
        }

        # [DaCe] setup temporaries
        self._temporaries = fvdyn_temporaries(
            quantity_factory, grid_indexing.domain_full(add=(1, 1, 1)), grid_data
        )
        state.__dict__.update(self._temporaries)

        # Build advection stencils
        self.tracer_advection = tracer_2d_1l.TracerAdvection(
            stencil_factory, tracer_transport, comm, self.tracers, grid_data
        )
        self._ak = ak
        self._bk = bk
        self._phis = phis.storage
        pfull_stencil = stencil_factory.from_origin_domain(
            init_pfull, origin=(0, 0, 0), domain=(1, 1, grid_indexing.domain[2])
        )
        pfull = utils.make_storage_from_shape(
            (1, 1, self._ak.shape[0]), is_temporary=False
        )
        pfull_stencil(self._ak, self._bk, pfull, self.config.p_ref)
        # workaround because cannot write to FieldK storage in stencil
        self._pfull = utils.make_storage_data(pfull[0, 0, :], self._ak.shape, (0,))
        self._fv_setup_stencil = stencil_factory.from_origin_domain(
            moist_cv.fv_setup,
            externals={
                "nwat": self.config.nwat,
                "moist_phys": self.config.moist_phys,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._pt_adjust_stencil = stencil_factory.from_origin_domain(
            pt_adjust,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._set_omega_stencil = stencil_factory.from_origin_domain(
            set_omega,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._copy_stencil = stencil_factory.from_origin_domain(
            copy_defn,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
        )
        self.acoustic_dynamics = AcousticDynamics(
            comm,
            stencil_factory,
            grid_data,
            damping_coefficients,
            config.grid_type,
            nested,
            stretched_grid,
            self.config.acoustic_dynamics,
            self._ak,
            self._bk,
            self._pfull,
            self._phis,
            state,
        )
        self._hyperdiffusion = HyperdiffusionDamping(
            stencil_factory,
            damping_coefficients,
            grid_data.rarea,
            self.config.nf_omega,
        )
        self._cubed_to_latlon = CubedToLatLon(
            state, stencil_factory, grid_data, order=config.c2l_ord, comm=comm
        )

        if not (not self.config.inline_q and NQ != 0):
            raise NotImplementedError("tracer_2d not implemented, turn on z_tracer")
        self._adjust_tracer_mixing_ratio = AdjustNegativeTracerMixingRatio(
            stencil_factory,
            self.config.check_negative,
            self.config.hydrostatic,
        )

        self._lagrangian_to_eulerian_obj = LagrangianToEulerian(
            stencil_factory,
            config.remapping,
            grid_data.area_64,
            NQ,
            self._pfull,
            tracers=self.tracers,
        )

        full_xyz_spec = grid_indexing.get_quantity_halo_spec(
            grid_indexing.domain_full(add=(1, 1, 1)),
            grid_indexing.origin_compute(),
            dims=[fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM],
            n_halo=utils.halo,
        )
        # [DaCe] Wrapping halo updater to a DaCe callback
        self._omega_halo_updater = AcousticDynamics._WrappedHaloUpdater(
            comm.get_scalar_halo_updater([full_xyz_spec]),
            state,
            ["omga_quantity"],
            comm=comm,
            grid=spec.grid,
        )

        # [DaCe] avoid parsing Timer as an argument
        self.timer = timer

    # [DaCe] Unroll all timers as callbacks to get around issues with parsing
    # context and parsing paramters of callbacks
    @dace_inhibitor
    def timer_start_remapping(self):
        self.timer.start("Remapping")

    @dace_inhibitor
    def timer_stop_remapping(self):
        self.timer.stop("Remapping")

    @dace_inhibitor
    def timer_start_dycore(self):
        self.timer.start("DynCore")

    @dace_inhibitor
    def timer_stop_dycore(self):
        self.timer.stop("DynCore")

    @dace_inhibitor
    def timer_start_tracers(self):
        self.timer.start("TracerAdvection")

    @dace_inhibitor
    def timer_stop_tracers(self):
        self.timer.stop("TracerAdvection")

    # [DaCe] new function allowing pos-constructor state update from caller code
    def update_state(
        self,
        conserve_total_energy,
        timestep,
        do_adiabatic_init,
        ptop,
        n_split,
        ks,
        state: SimpleNamespace,
    ):
        # [DaCe] Update state
        state.__dict__.update(
            {
                "consv_te": conserve_total_energy,
                "bdt": timestep,
                "mdt": timestep / self.config.k_split,
                "do_adiabatic_init": do_adiabatic_init,
                "ptop": ptop,
                "n_split": n_split,
                "k_split": self.config.k_split,
                "ks": ks,
            }
        )
        state.__dict__.update(self._temporaries)
        state.__dict__.update(self.acoustic_dynamics._temporaries)

    # [DaCe] move compute_preamble inside the class to go around an issue
    #       with passing stencils as a parameter
    @computepath_method
    def compute_preamble(self, state: dace_constant, is_root_rank: bool):
        if self.config.hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")
        if __debug__:
            if is_root_rank:
                print("FV Setup")
        self._fv_setup_stencil(
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qsnow,
            state.qice,
            state.qgraupel,
            state.q_con,
            state.cvm,
            state.pkz,
            state.pt,
            state.cappa,
            state.delp,
            state.delz,
            state.dp1,
        )

        if state.consv_te > 0 and not state.do_adiabatic_init:
            raise NotImplementedError(
                "compute total energy is not implemented, it needs an allReduce"
            )

        if (not self.config.rf_fast) and self.config.tau != 0:
            raise NotImplementedError(
                "Rayleigh_Super, called when rf_fast=False and tau !=0"
            )

        if self.config.adiabatic and self.config.kord_tm > 0:
            raise NotImplementedError(
                "unimplemented namelist options adiabatic with positive kord_tm"
            )
        else:
            if __debug__:
                if is_root_rank:
                    print("Adjust pt")
            self._pt_adjust_stencil(
                state.pkz,
                state.dp1,
                state.q_con,
                state.pt,
            )

    @computepath_method
    def __call__(self, *args, **kwargs):
        return self.step_dynamics(*args, **kwargs)

    # [DaCe] less parameters since update_state happens now in the __init__
    @computepath_method
    def step_dynamics(
        self,
        state: dace_constant,
    ):
        """
        Step the model state forward by one timestep.

        Args:
            state: model prognostic state and inputs
            conserve_total_energy: if True, conserve total energy
            do_adiabatic_init: if True, do adiabatic dynamics. Used
                for model initialization.
            timestep: time to progress forward in seconds
            ptop: pressure at top of atmosphere
            n_split: number of acoustic timesteps per remapping timestep
            ks: the lowest index (highest layer) for which rayleigh friction
                and other rayleigh computations are done
            timer: if given, use for timing model execution
        """
        cb_nvtx_range_push_dynsteps()
        # [DaCe] Updating states outside runtime path (__dict__.update)
        self._compute(state)
        cb_nvtx_range_pop()

    @computepath_method
    def _compute(self, state: dace_constant):
        # [DaCe] Move to update_state temporaries update

        # [Dace] Move the tracers setup to __init__

        # [DaCe] remove this code, it is not used anywhere in the dycore
        # state.ak = self._ak
        # state.bk = self._bk

        last_step = False
        self.compute_preamble(
            state,
            is_root_rank=self.comm_rank == 0,
        )

        # [DaCe] Do not unroll this top-level loop to contain compile time
        for k_split in dace_no_unroll(range(state.k_split)):
            # [DaCe] can't change the global state (declared constant), can't pass it down to acoustics substep (bad types at parsing dace.Scalar)
            # state.n_map = k_split + 1
            n_map = k_split + 1
            last_step = k_split == state.k_split - 1
            # [DaCe] unrolling the call of ._dyn which leads to MPI.comm being deep_copied (and failing)
            #        unclear why
            self._dyn(state=state, tracers=self.tracers, n_map=n_map)

            if self.grid_indexing.domain[2] > 4:
                # nq is actually given by ncnst - pnats,
                # where those are given in atmosphere.F90 by:
                # ncnst = Atm(mytile)%ncnst
                # pnats = Atm(mytile)%flagstruct%pnats
                # here we hard-coded it because 8 is the only supported value,
                # refactor this later!

                # do_omega = self.namelist.hydrostatic and last_step
                # TODO: Determine a better way to do this, polymorphic fields perhaps?
                # issue is that set_val in map_single expects a 3D field for the
                # "surface" array
                if __debug__:
                    if self.comm_rank == 0:
                        print("Remapping")
                # [DaCe] Context manager are unimplemented
                self.timer_start_remapping()
                self._lagrangian_to_eulerian_obj(
                    self.tracer_storages,
                    state.pt,
                    state.delp,
                    state.delz,
                    state.peln,
                    state.u,
                    state.v,
                    state.w,
                    state.ua,
                    state.va,
                    state.cappa,
                    state.q_con,
                    state.qcld,
                    state.pkz,
                    state.pk,
                    state.pe,
                    state.phis,
                    state.te0_2d,
                    state.ps,
                    state.wsd,
                    state.omga,
                    self._ak,
                    self._bk,
                    self._pfull,
                    state.dp1,
                    state.ptop,
                    constants.KAPPA,
                    constants.ZVIR,
                    last_step,
                    state.consv_te,
                    state.bdt / state.k_split,
                    state.bdt,
                    state.do_adiabatic_init,
                    NQ,
                )
                if last_step:
                    self.post_remap(
                        state,
                        is_root_rank=self.comm_rank == 0,
                        da_min=self._da_min,
                    )
                self.timer_stop_remapping()
        self.wrapup(
            state,
            is_root_rank=self.comm_rank == 0,
        )

    @computepath_method
    def _dyn(self, state: dace_constant, tracers: dace_constant, n_map):
        self._copy_stencil(
            state.delp,
            state.dp1,
        )
        if __debug__:
            if self.comm_rank == 0:
                print("DynCore")
        # [DaCe] context mananger fails parsing
        self.timer_start_dycore()
        self.acoustic_dynamics(
            state,
            n_map=n_map,
            update_temporaries=False,
        )
        self.timer_stop_dycore()
        if self.config.z_tracer:
            if __debug__:
                if self.comm_rank == 0:
                    print("TracerAdvection")
            # [DaCe] context mananger fails parsing
            self.timer_start_tracers()
            self.tracer_advection(
                tracers,
                state.dp1,
                state.mfxd,
                state.mfyd,
                state.cxd,
                state.cyd,
                state.mdt,
            )
            self.timer_stop_tracers()

    # [DaCe] moving post_remap inside the class to workaround an issue passing stencils as parameters
    @computepath_method
    def post_remap(
        self,
        state: dace_constant,
        is_root_rank: bool,
        da_min: FloatFieldIJ,
    ):
        if not self.config.hydrostatic:
            if __debug__:
                if is_root_rank:
                    print("Omega")
            self._set_omega_stencil(
                state.delp,
                state.delz,
                state.w,
                state.omga,
            )
        if self.config.nf_omega > 0:
            if __debug__:
                if is_root_rank == 0:
                    print("Del2Cubed")
            self._omega_halo_updater.update()
            self._hyperdiffusion(state.omga, 0.18 * da_min)

    # [DaCe] moving post_remap inside the class to workaround an issue passing stencils as parameters
    @computepath_method
    def wrapup(
        self,
        state: dace_constant,
        is_root_rank: bool,
    ):
        if __debug__:
            if is_root_rank:
                print("Neg Adj 3")
        self._adjust_tracer_mixing_ratio(
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qsnow,
            state.qice,
            state.qgraupel,
            state.qcld,
            state.pt,
            state.delp,
            state.delz,
            state.peln,
        )

        if __debug__:
            if is_root_rank:
                print("CubedToLatLon")
        self._cubed_to_latlon(
            state.u,
            state.v,
            state.ua,
            state.va,
        )
