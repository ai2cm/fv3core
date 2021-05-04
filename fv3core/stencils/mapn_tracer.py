from typing import Dict

from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.fillz import FillNegativeTracerValues
from fv3core.stencils.map_single import MapSingle
from fv3core.stencils.remap_profile import RemapProfile
from fv3core.utils.typing import FloatField


@gtstencil()
def set_components(
    tracer: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
):
    with computation(PARALLEL), interval(...):
        a4_1 = tracer
        a4_2 = 0.0
        a4_3 = 0.0
        a4_4 = 0.0


def compute(
    pe1: FloatField,
    pe2: FloatField,
    dp2: FloatField,
    tracers: Dict[str, "FloatField"],
    nq: int,
    q_min: float,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    kord: int,
):
    remapping_calculation = utils.cached_stencil_class(RemapProfile)(
        kord, 0, i1, i2, j1, j2, cache_key=f"map_profile_{kord}_0"
    )

    domain_compute = (
        spec.grid.ie - spec.grid.is_ + 1,
        spec.grid.je - spec.grid.js + 1,
        spec.grid.npz + 1,
    )

    qs = utils.make_storage_from_shape(
        pe1.shape, origin=(0, 0, 0), cache_key="mapn_tracer_qs"
    )

    map_single = utils.cached_stencil_class(MapSingle)(
        kord, 0, i1, i2, j1, j2, cache_key="mapntracer-single"
    )
    tracer_list = [tracers[q] for q in utils.tracer_variables[0:nq]]
    map_single.setup_data(tracer_list[0], pe1)

    for tracer in tracer_list:
        set_components(
            tracer,
            map_single.q4_1,
            map_single.q4_2,
            map_single.q4_3,
            map_single.q4_4,
            origin=(spec.grid.is_, spec.grid.js, 0),
            domain=domain_compute,
        )
        q4_1, q4_2, q4_3, q4_4 = remapping_calculation(
            qs,
            map_single.q4_1,
            map_single.q4_2,
            map_single.q4_3,
            map_single.q4_4,
            map_single.dp1,
            q_min,
        )
        map_single.lagrangian_contributions(
            tracer,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            map_single.dp1,
        )
    if spec.namelist.fill:
        fillz = utils.cached_stencil_class(FillNegativeTracerValues)(
            cache_key="mapntracer-fillz"
        )
        fillz(dp2, tracers, map_single.i_extent, map_single.j_extent, spec.grid.npz, nq)
