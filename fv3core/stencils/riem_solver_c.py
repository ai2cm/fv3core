import typing

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.sim1_solver import Sim1Solver
from fv3core.utils.typing import FloatField, FloatFieldIJ


@typing.no_type_check
def precompute(
    delpc: FloatField,
    cappa: FloatField,
    w3: FloatField,
    w: FloatField,
    cp3: FloatField,
    gz: FloatField,
    dm: FloatField,
    q_con: FloatField,
    pem: FloatField,
    dz: FloatField,  # is actually delta of gz
    gm: FloatField,
    pm: FloatField,
    ptop: float,
):
    with computation(PARALLEL), interval(...):
        dm = delpc
        cp3 = cappa
        w = w3
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peg = ptop
        with interval(1, None):
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
    with computation(PARALLEL), interval(0, -1):
        dz = gz[0, 0, 1] - gz
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm / constants.GRAV
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / log(peg[0, 0, 1] / peg)


def finalize(
    pe2: FloatField,
    pem: FloatField,
    hs: FloatFieldIJ,
    dz: FloatField,
    pef: FloatField,
    gz: FloatField,
    ptop: float,
):
    # TODO: We only want to bottom level of hd, so this could be removed once
    # hd0 is a 2d field.
    with computation(FORWARD):
        with interval(0, 1):
            hs_0 = hs
        with interval(1, None):
            hs_0 = hs_0[0, 0, -1]
    with computation(PARALLEL):
        with interval(0, 1):
            pef = ptop
        with interval(1, None):
            pef = pe2 + pem
    with computation(BACKWARD):
        with interval(-1, None):
            gz = hs_0
        with interval(0, -1):
            gz = gz[0, 0, 1] - dz * constants.GRAV


class RiemannSolverC:
    """
    Fortran subroutine Riem_Solver_C
    """

    def __init__(self, namelist):
        grid = spec.grid
        is1 = grid.is_ - 1
        ie1 = grid.ie + 1
        js1 = grid.js - 1
        je1 = grid.je + 1
        km = spec.grid.npz - 1
        riemorigin = (is1, js1, 0)
        shape = w3.shape

        domain = (grid.nic + 2, grid.njc + 2, km + 2)

        self._tmp_dm = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_cp3 = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_w = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pem = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pe = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_gm = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_dz = utils.make_storage_from_shape(shape, riemorigin)
        self._tmp_pm = utils.make_storage_from_shape(shape, riemorigin)

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
        self._sim1_solve = Sim1Solver(
            namelist,
            grid,
            is1,
            ie1,
            js1,
            je1,
        )

    def __call__(
        self,
        ms: int,
        dt2: float,
        akap: float,
        cappa: FloatField,
        ptop: float,
        hs: FloatFieldIJ,
        w3: FloatField,
        ptc: FloatField,
        q_con: FloatField,
        delpc: FloatField,
        gz: FloatField,
        pef: FloatField,
        ws: FloatFieldIJ,
    ):
        self._precompute_stencil(
            delpc,
            cappa,
            w3,
            w,
            cp3,
            gz,
            dm,
            q_con,
            pem,
            dz,
            gm,
            pm,
            ptop,
        )


        self._sim1_solve(dt2, gm, cp3, pe, dm, pm, pem, w, dz, ptc, ws)

        self._finalize_stencil(pe, pem, hs, dz, pef, gz, ptop)


# TODO: this is totally inefficient, can we use stencils?
def compute(
    ms: int,
    dt2: float,
    akap: float,
    cappa: FloatField,
    ptop: float,
    hs: FloatFieldIJ,
    w3: FloatField,
    ptc: FloatField,
    q_con: FloatField,
    delpc: FloatField,
    gz: FloatField,
    pef: FloatField,
    ws: FloatFieldIJ,
):
    grid = spec.grid
    is1 = grid.is_ - 1
    ie1 = grid.ie + 1
    js1 = grid.js - 1
    je1 = grid.je + 1
    km = spec.grid.npz - 1
    shape = w3.shape
    domain = (spec.grid.nic + 2, grid.njc + 2, km + 2)
    riemorigin = (is1, js1, 0)
    dm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riemc_dm")
    cp3 = utils.make_storage_from_shape(shape, riemorigin, cache_key="riemc_cp3")
    w = utils.make_storage_from_shape(shape, riemorigin, cache_key="riemc_w")
    pem = utils.make_storage_from_shape(
        shape, riemorigin, cache_key="riem_solver_c_pem"
    )
    pe = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver_c_pe")
    gm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver_c_gm")
    dz = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver_c_dz")
    pm = utils.make_storage_from_shape(shape, riemorigin, cache_key="riem_solver_c_pm")
    # it looks like this code sets pef = ptop, and does not otherwise use pef here
    precompute(
        delpc,
        cappa,
        w3,
        w,
        cp3,
        gz,
        dm,
        q_con,
        pem,
        dz,
        gm,
        pm,
        ptop,
        origin=riemorigin,
        domain=domain,
    )

    sim1_solve = utils.cached_stencil_class(Sim1Solver)(
        spec.namelist,
        spec.grid,
        is1,
        ie1,
        js1,
        je1,
        cache_key="riem_solver_c_sim1solver",
    )
    sim1_solve(dt2, gm, cp3, pe, dm, pm, pem, w, dz, ptc, ws)

    finalize(pe, pem, hs, dz, pef, gz, ptop, origin=riemorigin, domain=domain)
