import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.fillz as fillz
import fv3core.stencils.map_single as map_single
import fv3core.stencils.remap_profile as remap_profile
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.basic_operations import copy_stencil


def compute(
    pe1, pe2, dp2, tracers, nq, q_min, i1, i2, kord, j_2d=None, version="stencil",  # "transliterated"
):
    grid = spec.grid
    fill = spec.namelist.fill
    qs = utils.make_storage_from_shape(pe1.shape, origin=(0, 0, 0))
    (
        dp1,
        q4_1,
        q4_2,
        q4_3,
        q4_4,
        origin,
        domain,
        jslice,
        i_extent,
    ) = map_single.setup_data(tracers[utils.tracer_variables[0]], pe1, i1, i2, j_2d)

    tracer_list = [tracers[q] for q in utils.tracer_variables[0:nq]]
    # transliterated fortran 3d or 2d validate, not bit-for bit
    # for q in utils.tracer_variables[0:nq]:
    for tracer in tracer_list:
        # if j_2d is None:
        copy_stencil(tracer, q4_1, origin=origin, domain=domain)
        # else:
        #    q4_1.data[:] = tracer.data[:]
        q4_2[:] = 0.0
        q4_3[:] = 0.0
        q4_4[:] = 0.0
        q4_1, q4_2, q4_3, q4_4 = remap_profile.compute(
            qs, q4_1, q4_2, q4_3, q4_4, dp1, grid.npz, i1, i2, 0, kord, jslice, q_min
        )
        map_single.do_lagrangian_contributions(
            tracer,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            i1,
            i2,
            kord,
            jslice,
            origin,
            domain,
            version,
        )
    if fill:
        fillz.compute(dp2, tracers, i_extent, spec.grid.npz, nq, jslice)
