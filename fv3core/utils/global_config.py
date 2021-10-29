import functools
import os
from typing import Optional

from .profiler import (
    BaseProfiler,
    CPUProfiler,
    CUDAProfiler,
    NoneProfiler,
    ProfileLevel,
)


def getenv_bool(name: str, default: str) -> bool:
    indicator = os.getenv(name, default).title()
    return indicator == "True"


def set_backend(new_backend: str):
    global _BACKEND
    _BACKEND = new_backend
    for function in (is_gpu_backend, is_gtc_backend):
        if hasattr(function, "cache_clear"):
            function.cache_clear()
    set_profiler()


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


def init_profile_level():
    getenv_bool("FV3_PROFILE", "False")
    return ProfileLevel(int(os.getenv("FV3_PROFILE", 0)))


def get_profile_level() -> ProfileLevel:
    return _PROFILE_LEVEL


def get_profiler():
    return _PROFILER


def set_profiler():
    global _PROFILER
    if get_profile_level() >= ProfileLevel.TIMINGS:
        if is_gpu_backend():
            _PROFILER = CUDAProfiler()
        else:
            _PROFILER = CPUProfiler()


# Options: numpy, gtx86, gtcuda, debug
_BACKEND: Optional[str] = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD: bool = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_VALIDATE_ARGS: bool = True
# Profile level decide of the granularity of the profiler
# Initialize by reading FV3_PROFILE=X with X an int of the level asked
_PROFILE_LEVEL: ProfileLevel = init_profile_level()
# Global profiler, defaulting to a no-op profiler, to be used
# throughout the code
_PROFILER: BaseProfiler = NoneProfiler()
