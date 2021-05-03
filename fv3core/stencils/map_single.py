from types import SimpleNamespace
from typing import List, Tuple

from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import StencilWrapper
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.remap_profile import RemapProfile
from fv3core.utils.typing import FloatField, FloatFieldIJ


def set_dp(dp1: FloatField, pe1: FloatField):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


def set_eulerian_pressures(pe: FloatField, ptop: FloatFieldIJ, pbot: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        ptop = pe[0, 0, 0]
        pbot = pe[0, 0, 1]


def set_remapped_quantity(q: FloatField, set_values: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        q = set_values[0, 0]


def lagrangian_contributions(
    pe1: FloatField,
    ptop: FloatFieldIJ,
    pbot: FloatFieldIJ,
    q4_1: FloatField,
    q4_2: FloatField,
    q4_3: FloatField,
    q4_4: FloatField,
    dp1: FloatField,
    q2_adds: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        q2_tmp = 0.0
        if pe1 < pbot and pe1[0, 0, 1] > ptop:
            # We are in the right pressure range to contribute to the Eulerian cell
            if pe1 <= ptop:
                # we are in the first Lagrangian level that conributes
                pl = (ptop - pe1) / dp1
                if pbot <= pe1[0, 0, 1]:
                    # Eulerian grid element is contained in the Lagrangian element
                    pr = (pbot - pe1) / dp1
                    q2_tmp = (
                        q4_2
                        + 0.5 * (q4_4 + q4_3 - q4_2) * (pr + pl)
                        - q4_4 * (1.0 / 3.0) * (pr * (pr + pl) + pl ** 2)
                    )
                else:
                    # Eulerian element encompasses multiple Lagrangian elements
                    # and this is just the first one
                    q2_tmp = (
                        (pe1[0, 0, 1] - ptop)
                        * (
                            q4_2
                            + 0.5 * (q4_4 + q4_3 - q4_2) * (1.0 + pl)
                            - q4_4 * (1.0 / 3.0) * (1.0 + pl * (1.0 + pl))
                        )
                        / (pbot - ptop)
                    )
            else:
                # we are in a farther-down level
                if pbot > pe1[0, 0, 1]:
                    # add the whole level to the Eulerian cell
                    q2_tmp = dp1 * q4_1 / (pbot - ptop)
                else:
                    # this is the bottom layer that contributes
                    dp = pbot - pe1
                    esl = dp / dp1
                    q2_tmp = (
                        dp
                        * (
                            q4_2
                            + 0.5
                            * esl
                            * (q4_3 - q4_2 + q4_4 * (1.0 - (2.0 / 3.0) * esl))
                        )
                        / (pbot - ptop)
                    )
    with computation(FORWARD), interval(0, 1):
        q2_adds = 0.0
    with computation(FORWARD), interval(...):
        q2_adds += q2_tmp


class LagrangianContributions:
    """
    Fortran name is lagrangian_contributions
    """

    def __init__(self):
        self._grid = spec.grid
        self._km = self._grid.npz
        shape = self._grid.domain_shape_full(add=(1, 1, 1))
        shape2d = shape[0:2]

        self._q2_adds = utils.make_storage_from_shape(shape2d)
        self._ptop = utils.make_storage_from_shape(shape2d)
        self._pbot = utils.make_storage_from_shape(shape2d)

        self._set_eulerian_pressures = StencilWrapper(set_eulerian_pressures)
        self._lagrangian_contributions = StencilWrapper(lagrangian_contributions)
        self._set_remapped_quantity = StencilWrapper(set_remapped_quantity)

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        q4_1: FloatField,
        q4_2: FloatField,
        q4_3: FloatField,
        q4_4: FloatField,
        dp1: FloatField,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        origin: Tuple[int, int, int],
        domain: Tuple[int, int, int],
    ):
        # A stencil with a loop over k2:
        for k_eul in range(self._km):
            # TODO (olivere): This is hacky
            # merge with subsequent stencil when possible
            self._set_eulerian_pressures(
                pe2,
                self._ptop,
                self._pbot,
                origin=(origin[0], origin[1], k_eul),
                domain=(domain[0], domain[1], 1),
            )

            self._lagrangian_contributions(
                pe1,
                self._ptop,
                self._pbot,
                q4_1,
                q4_2,
                q4_3,
                q4_4,
                dp1,
                self._q2_adds,
                origin=origin,
                domain=domain,
            )

            self._set_remapped_quantity(
                q1,
                self._q2_adds,
                origin=(origin[0], origin[1], k_eul),
                domain=(domain[0], domain[1], 1),
            )


class MapSingle:
    """
    Fortran name is map_single, test class is Map1_PPM_2d
    """

    def __init__(self, namelist: SimpleNamespace):
        self._grid = spec.grid
        shape = self._grid.domain_shape_full(add=(1, 1, 1))
        origin = self._grid.compute_origin()

        self.dp1 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_1 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_2 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_3 = utils.make_storage_from_shape(shape, origin=origin)
        self.q4_4 = utils.make_storage_from_shape(shape, origin=origin)

        self.lagrangian_contributions = LagrangianContributions()

        self._used_kords: Tuple[int] = (namelist.kord_tm, namelist.kord_tr)
        self._used_modes: Tuple[int] = (-2, -1, 1)

        kord_mode_combos: List[Tuple[int]] = []
        for kord in self._used_kords:
            for mode in self._used_modes:
                kord_mode_combos.append((kord, mode))

        self._remap_profiles = {pair: RemapProfile(*pair) for pair in kord_mode_combos}
        self._set_dp = StencilWrapper(set_dp)

    def __call__(
        self,
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
    ):
        """
        Compute x-flux using the PPM method.

        Args:
            q1 (in): Transported scalar
            pe1 (in): ???
            pe2 (out): ???
            qs (out): ???
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        self.setup_data(q1, pe1, i1, i2, j1, j2)
        q4_1, q4_2, q4_3, q4_4 = self._remap_profiles[(kord, mode)](
            qs,
            self.q4_1,
            self.q4_2,
            self.q4_3,
            self.q4_4,
            self.dp1,
            i1,
            i2,
            j1,
            j2,
            qmin,
        )
        self.lagrangian_contributions(
            q1,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            self.dp1,
            i1,
            i2,
            j1,
            j2,
            self.origin,
            self.domain,
        )
        return q1

    def setup_data(
        self, q1: FloatField, pe1: FloatField, i1: int, i2: int, j1: int, j2: int
    ):
        grid = self._grid
        self.i_extent = i2 - i1 + 1
        self.j_extent = j2 - j1 + 1
        self.origin = (i1, j1, 0)
        self.domain = (self.i_extent, self.j_extent, grid.npz)

        copy_stencil(
            q1,
            self.q4_1,
            origin=(0, 0, 0),
            domain=grid.domain_shape_full(),
        )
        self._set_dp(self.dp1, pe1, origin=self.origin, domain=self.domain)
