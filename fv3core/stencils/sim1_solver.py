import typing

from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, exp, interval, log

import fv3core.utils.global_constants as constants
from fv3core.decorators import StencilWrapper
from fv3core.utils.typing import FloatField, FloatFieldIJ


@typing.no_type_check
def sim1_solver(
    w: FloatField,
    dm: FloatField,
    gm: FloatField,
    dz: FloatField,
    ptr: FloatField,
    pm: FloatField,
    pe: FloatField,
    pem: FloatField,
    wsr: FloatFieldIJ,
    cp3: FloatField,
    dt: float,
    t1g: float,
    rdt: float,
    p_fac: float,
):
    # TODO: We only want to bottom level of wsr, so this could be removed once
    # wsr_top is a 2d field.
    with computation(FORWARD):
        with interval(0, 1):
            wsr_top = wsr
        with interval(1, None):
            wsr_top = wsr_top[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        pe = exp(gm * log(-dm / dz * constants.RDGAS * ptr)) - pm
        w1 = w
    with computation(FORWARD):
        with interval(0, -2):
            g_rat = dm / dm[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe + g_rat * pe[0, 0, 1])
        with interval(-2, -1):
            bb = 2.0
            dd = 3.0 * pe
    # bet[i,j,k] = bb[i,j,0]
    with computation(FORWARD):
        with interval(0, 1):
            bet = bb
        with interval(1, -1):
            bet = bet[0, 0, -1]

    # stencils: w_solver
    # {
    with computation(PARALLEL):
        with interval(0, 1):
            pp = 0.0
        with interval(1, 2):
            pp = dd[0, 0, -1] / bet
    with computation(FORWARD), interval(1, -1):
        gam = g_rat[0, 0, -1] / bet[0, 0, -1]
        bet = bb - gam
    with computation(FORWARD), interval(2, None):
        pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0, -1]
    with computation(BACKWARD), interval(1, -1):
        pp = pp - gam * pp[0, 0, 1]
        # w solver
        aa = t1g * 0.5 * (gm[0, 0, -1] + gm) / (dz[0, 0, -1] + dz) * (pem + pp)
    # }
    # updates on bet:
    with computation(FORWARD):
        with interval(0, 1):
            bet = dm[0, 0, 0] - aa[0, 0, 1]
        with interval(1, None):
            bet = bet[0, 0, -1]
    # w_pe_dz_compute
    # {
    with computation(FORWARD):
        with interval(0, 1):
            w = (dm * w1 + dt * pp[0, 0, 1]) / bet
        with interval(1, -2):
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + aa[0, 0, 1] + aa * gam)
            w = (dm * w1 + dt * (pp[0, 0, 1] - pp) - aa * w[0, 0, -1]) / bet
        with interval(-2, -1):
            p1 = t1g * gm / dz * (pem[0, 0, 1] + pp[0, 0, 1])
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + p1 + aa * gam)
            w = (
                dm * w1 + dt * (pp[0, 0, 1] - pp) - p1 * wsr_top - aa * w[0, 0, -1]
            ) / bet
    with computation(BACKWARD), interval(0, -2):
        w = w - gam[0, 0, 1] * w[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe = 0.0
        with interval(1, None):
            pe = pe[0, 0, -1] + dm[0, 0, -1] * (w[0, 0, -1] - w1[0, 0, -1]) * rdt
    with computation(BACKWARD):
        with interval(-2, -1):
            p1 = (pe + 2.0 * pe[0, 0, 1]) * 1.0 / 3.0
        with interval(0, -2):
            p1 = (pe + bb * pe[0, 0, 1] + g_rat * pe[0, 0, 2]) * 1.0 / 3.0 - g_rat * p1[
                0, 0, 1
            ]
    with computation(PARALLEL), interval(0, -1):
        maxp = p_fac * pm if p_fac * dm > p1 + pm else p1 + pm
        dz = -dm * constants.RDGAS * ptr * exp((cp3 - 1.0) * log(maxp))
    # }


class Sim1Solver:
    """
    Fortran name is sim1_solver

    Namelist:
        p_fac: Safety factor for minimum nonhydrostatic pressures.
    """

    # TODO: implement MOIST_CAPPA=false

    def __init__(self, namelist, grid, istart, iend, jstart, jend):
        self._pfac = namelist.p_fac
        nic = iend - istart + 1
        njc = jend - jstart + 1
        self._compute_sim1_solve = StencilWrapper(
            func=sim1_solver,
            origin=(istart, jstart, 0),
            domain=(nic, njc, grid.npz + 1),
        )

    def __call__(
        self,
        dt: float,
        gm: FloatField,
        cp3: FloatField,
        pe: FloatField,
        dm: FloatField,
        pm: FloatField,
        pem: FloatField,
        w: FloatField,
        dz: FloatField,
        ptr: FloatField,
        wsr: FloatFieldIJ,
    ):
        t1g = 2.0 * dt * dt
        rdt = 1.0 / dt
        self._compute_sim1_solve(
            w,
            dm,
            gm,
            dz,
            ptr,
            pm,
            pe,
            pem,
            wsr,
            cp3,
            dt,
            t1g,
            rdt,
            self._pfac,
        )
