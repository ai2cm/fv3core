import functools
import os
from typing import Callable, Optional

import fv3gfs.util as util


def getenv_bool(name: str, default: str) -> bool:
    indicator = os.getenv(name, default).title()
    return indicator == "True"


def set_backend(new_backend: str):
    global _BACKEND
    _BACKEND = new_backend
    for function in (is_gpu_backend, is_gtc_backend):
        function.cache_clear()


def get_backend() -> str:
    return _BACKEND


def set_rebuild(flag: bool):
    global _REBUILD
    _REBUILD = flag


def get_rebuild() -> bool:
    return _REBUILD


def set_validate_args(new_validate_args: bool):
    global _VALIDATE_ARGS
    _VALIDATE_ARGS = new_validate_args


# Set to "False" to skip validating gt4py stencil arguments
@functools.lru_cache(maxsize=None)
def get_validate_args() -> bool:
    return _VALIDATE_ARGS


@functools.lru_cache(maxsize=None)
def is_gpu_backend() -> bool:
    return get_backend().endswith("cuda") or get_backend().endswith("gpu")


@functools.lru_cache(maxsize=None)
def is_gtc_backend() -> bool:
    return get_backend().startswith("gtc")


# [DaCe] See description below. Dangerous, should be refactored out
# Either we can JIT properly via GTC or the compute path need to be able
# to trigger compilation at call time properly if you haven't anyone above you
def is_dacemode_codegen_whitelisted(func: Callable[..., None]) -> bool:
    """Whitelist of stencil function that need code generation in DACE mode.
    Some stencils are called within the __init__ and therefore will need to
    be pre-compiled nonetheless.
    """
    whitelist = [
        "dp_ref_compute",
        "cubic_spline_interpolation_constants",
        "calc_damp",
        "set_gz",
        "set_pem",
        "copy_defn",
        "compute_geopotential",
        # DynamicalCore
        "init_pfull",
        # CubedToLatLon for Metric/Grid/State calculation see dynamics
        "ord4_transform",
        "c2l_ord2",
        # Expanded grid variable
        "compute_coriolis_parameter_defn",
    ]
    return any(func.__name__ in name for name in whitelist)


# Options: numpy, gtx86, gtcuda, debug
_BACKEND: Optional[str] = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD: bool = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_VALIDATE_ARGS: bool = True


import enum


class DaCeOrchestration(enum.Enum):
    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


def load_dace_orchestration() -> DaCeOrchestration:
    return DaCeOrchestration[os.getenv("FV3_DACEMODE", "Python")]


def get_dacemode() -> DaCeOrchestration:
    global _DACEMODE
    return _DACEMODE


def is_dace_orchestrated() -> bool:
    return _DACEMODE != DaCeOrchestration.Python


def set_dacemode(dacemode: DaCeOrchestration):
    global _DACEMODE
    _DACEMODE = dacemode


# Python: python orchestration
# Build: compile & save SDFG only
# BuildAndRun: compile & save SDFG, then run
# Run: load from .so and run, will fail if .so is not available
_DACEMODE: DaCeOrchestration = load_dace_orchestration()


def get_partitioner() -> Optional[util.CubedSpherePartitioner]:
    print("partitioner is used")
    global _PARTITIONER
    return _PARTITIONER


def set_partitioner(partitioner: Optional[util.CubedSpherePartitioner]) -> None:
    global _PARTITIONER
    if _PARTITIONER is not None:
        print("re-setting the partitioner, why is that?")
    _PARTITIONER = partitioner
    print("partitioner is set")


def set_partitioner_once(partitioner: Optional[util.CubedSpherePartitioner]) -> None:
    global _PARTITIONER
    if _PARTITIONER is not None:
        _PARTITIONER = partitioner
        print("partitioner is set")


# Partitioner from fv3core
_PARTITIONER: Optional[util.CubedSpherePartitioner] = None
