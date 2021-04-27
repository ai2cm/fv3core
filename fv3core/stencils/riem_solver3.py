import math
import typing

from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import StencilWrapper
from fv3core.stencils.sim1_solver import Sim1Solver
from fv3core.utils.typing import FloatField, FloatFieldIJ


@typing.no_type_check
def precompute(
    delp: FloatField,
    cappa: FloatField,
    pe: FloatField,
    pe_init: FloatField,
    cp3: FloatField,
    dm: FloatField,
    zh: FloatField,
    q_con: FloatField,
    pem: FloatField,
    peln: FloatField,
    pk3: FloatField,
    gm: FloatField,
    dz: FloatField,
    pm: FloatField,
    ptop: float,
    peln1: float,
    ptk: float,
):
    with computation(PARALLEL), interval(...):
        dm = delp
        cp3 = cappa
        pe_init = pe
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peln = peln1
            pk3 = ptk
            peg = ptop
            pelng = peln1
        with interval(1, None):
            # TODO consolidate with riem_solver_c, same functions, math functions
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peln = log(pem)
            # Excluding contribution from condensates
            # peln used during remap; pk3 used only for p_grad
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pelng = log(peg)
            # interface pk is using constant akap
            pk3 = exp(constants.KAPPA * peln)
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm * constants.RGRAV
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)
        dz = zh[0, 0, 1] - zh


def finalize(
    zs: FloatFieldIJ,
    dz: FloatField,
    zh: FloatField,
    peln_run: FloatField,
    peln: FloatField,
    pk3: FloatField,
    pk: FloatField,
    pem: FloatField,
    pe: FloatField,
    ppe: FloatField,
    pe_init: FloatField,
    last_call: bool,
):
    with computation(PARALLEL), interval(...):
        if __INLINED(spec.namelist.use_logp):
            pk3 = peln_run
        if __INLINED(spec.namelist.beta < -0.1):
            ppe = pe + pem
        else:
            ppe = pe
        if last_call:
            peln = peln_run
            pk = pk3
            pe = pem
        else:
            pe = pe_init
    with computation(BACKWARD):
        with interval(-1, None):
            zh = zs
        with interval(0, -1):
            zh = zh[0, 0, 1] - dz


class RiemannSolver3:
    """
    Fortran subroutine Riem_Solver3
    """

    def __init__(self, namelist):
        grid = spec.grid
        self._sim1_solve = Sim1Solver(
            namelist,
            spec.grid,
            grid.is_,
            grid.ie,
            grid.js,
            grid.je,
        )
        riemorigin = grid.compute_origin()
        domain = grid.domain_shape_compute(add=(0, 0, 1))
        shape = grid.domain_shape_full(add=(1, 1, 1))
        self._tmp_dm = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_cp3 = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pe_init = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pm = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pem = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_peln_run = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_gm = utils.make_storage_from_shape(shape, riemorigin)
        self._precompute_stencil = StencilWrapper(
            precompute,
            origin=riemorigin,
            domain=domain,
        )
        self._finalize_stencil = StencilWrapper(
            finalize,
            origin=riemorigin,
            domain=domain,
        )

    def __call__(
        self,
        last_call: bool,
        dt: float,
        cappa: FloatField,
        ptop: float,
        zs: FloatFieldIJ,
        wsd: FloatField,
        delz: FloatField,
        q_con: FloatField,
        delp: FloatField,
        pt: FloatField,
        zh: FloatField,
        pe: FloatField,
        ppe: FloatField,
        pk3: FloatField,
        pk: FloatField,
        peln: FloatField,
        w: FloatFieldIJ,
    ):
        """
        Riemann solver for after D-grid winds advected and model heights updated,
        that accounts for vertically propagating sound waves by solving the
        nonhydrostatic terms for vertical velocity (w) and non-hydrostatic
        pressure perturbation.

        Args:
           last_call: boolean, is last acoustic timestep (in)
           dt: acoustic timestep in seconds (in)
           cappa: (in)
           ptop: pressure at top of atmosphere (in)
           zs: surface geopotential height(in)
           wsd: vertical velocity of the lowest level (in)
           delz: vertical delta of atmospheric layer in meters (in)
           q_con: total condensate mixing ratio (in)
           delp: vertical delta in pressure (in)
           pt: potential temperature (in)
           zh: geopotential heigh (inout)
           pe: full hydrostatic pressure(inout)
           ppe: non-hydrostatic pressure perturbation (inout)
           pk3: interface pressure raised to power of kappa using constant kappa (inout)
           pk: interface pressure raised to power of kappa, final acoustic value (inout)
           peln: logarithm of interface pressure(inout)
           w: vertical velocity (inout)
        """

        peln1 = math.log(ptop)
        ptk = math.exp(constants.KAPPA * peln1)
        self._precompute_stencil(
            delp,
            cappa,
            pe,
            self._tmp_pe_init,
            self._tmp_cp3,
            self._tmp_dm,
            zh,
            q_con,
            self._tmp_pem,
            self._tmp_peln_run,
            pk3,
            self._tmp_gm,
            delz,
            self._tmp_pm,
            ptop,
            peln1,
            ptk,
        )

        self._sim1_solve(
            dt,
            self._tmp_gm,
            self._tmp_cp3,
            pe,
            self._tmp_dm,
            self._tmp_pm,
            self._tmp_pem,
            w,
            delz,
            pt,
            wsd,
        )

        self._finalize_stencil(
            zs,
            delz,
            zh,
            self._tmp_peln_run,
            peln,
            pk3,
            pk,
            self._tmp_pem,
            pe,
            ppe,
            self._tmp_pe_init,
            last_call,
        )
