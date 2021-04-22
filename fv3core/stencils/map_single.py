from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.remap_profile import RemapProfile
from fv3core.utils.typing import FloatField, IntFieldIJ


r3 = 1.0 / 3.0
r23 = 2.0 / 3.0


@gtstencil()
def set_dp(dp1: FloatField, pe1: FloatField):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


def compute(
    q1: FloatField,
    pe1: FloatField,
    pe2: FloatField,
    qs: FloatField,
    mode: int,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    kord: int,
    qmin: float = 0.0,
    version: str = "stencil",
):
    remap_profile_k = utils.cached_stencil_class(RemapProfile)(
        kord, mode, cache_key=f"map_profile_{kord}_{mode}"
    )
    dp1, q4_1, q4_2, q4_3, q4_4, origin, domain, i_extent, j_extent = setup_data(
        q1, pe1, i1, i2, j1, j2
    )
    q4_1, q4_2, q4_3, q4_4 = remap_profile_k(
        qs, q4_1, q4_2, q4_3, q4_4, dp1, i1, i2, j1, j2, qmin
    )

    lev = utils.make_storage_from_shape(
        q1.shape[:-1],
        origin=spec.grid.compute_origin()[:-1],
        cache_key="lagrangian_contributions_lev",
        init=True,
        mask=(True, True, False),
        dtype=int,
    )

    lagrangian_contributions(
        q1,
        pe1,
        pe2,
        q4_1,
        q4_2,
        q4_3,
        q4_4,
        dp1,
        lev,
        origin=origin,
        domain=domain,
    )

    return q1


def setup_data(q1: FloatField, pe1: FloatField, i1: int, i2: int, j1: int, j2: int):
    grid = spec.grid
    i_extent = i2 - i1 + 1
    j_extent = j2 - j1 + 1
    origin = (i1, j1, 0)
    domain = (i_extent, j_extent, grid.npz)

    dp1 = utils.make_storage_from_shape(
        q1.shape, origin=origin, cache_key="map_single_dp1"
    )
    q4_1 = utils.make_storage_from_shape(
        q1.shape, origin=(grid.is_, 0, 0), cache_key="map_single_q4_1"
    )
    q4_2 = utils.make_storage_from_shape(
        q4_1.shape, origin=grid.compute_origin(), cache_key="map_single_q4_2"
    )
    q4_3 = utils.make_storage_from_shape(
        q4_1.shape, origin=grid.compute_origin(), cache_key="map_single_q4_3"
    )
    q4_4 = utils.make_storage_from_shape(
        q4_1.shape, origin=grid.compute_origin(), cache_key="map_single_q4_4"
    )
    copy_stencil(
        q1,
        q4_1,
        origin=(0, 0, 0),
        domain=grid.domain_shape_full(),
    )
    set_dp(dp1, pe1, origin=origin, domain=domain)
    return dp1, q4_1, q4_2, q4_3, q4_4, origin, domain, i_extent, j_extent


@gtstencil()
def lagrangian_contributions(
    q: FloatField,
    pe1: FloatField,
    pe2: FloatField,
    q4_1: FloatField,
    q4_2: FloatField,
    q4_3: FloatField,
    q4_4: FloatField,
    dp1: FloatField,
    lev: IntFieldIJ,
):
    with computation(FORWARD), interval(...):
        v_pe2 = pe2
        v_pe1 = pe1[0, 0, lev]
        pl = (v_pe2 - v_pe1) / dp1[0, 0, lev]
        if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
            pr = (pe2[0, 0, 1] - v_pe1) / dp1[0, 0, lev]
            q = (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (pr + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
            )
        else:
            qsum = (pe1[0, 0, lev + 1] - pe2) * (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (1.0 + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
            )
            lev = lev + 1
            while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                lev = lev + 1
            dp = pe2[0, 0, 1] - pe1[0, 0, lev]
            esl = dp / dp1[0, 0, lev]
            qsum += dp * (
                q4_2[0, 0, lev]
                + 0.5
                * esl
                * (
                    q4_3[0, 0, lev]
                    - q4_2[0, 0, lev]
                    + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                )
            )
            q = qsum / (pe2[0, 0, 1] - pe2)
        lev = lev - 1
