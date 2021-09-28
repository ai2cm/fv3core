import hashlib
import os
from collections.abc import Hashable
from typing import Any, Callable, Dict


_backend_options: Dict[str, Any] = {
    "all": {
        "skip_passes": ("graph_merge_horizontal_executions",),
        "use_buffer_interface": True,
    },
    "fv_subgridz.init": {
        "skip_passes": ("KCacheDetection",),
    },
    "ytp_v._ytp_v": {
        "skip_passes": ("GreedyMerging",),
    },
}


def getenv_bool(name: str, default: str) -> bool:
    indicator = os.getenv(name, default).title()
    return indicator == "True"


def set_backend(new_backend: str):
    global _BACKEND
    _BACKEND = new_backend


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
def get_validate_args() -> bool:
    return _VALIDATE_ARGS


def set_format_source(flag: bool):
    global _FORMAT_SOURCE
    _FORMAT_SOURCE = flag


def get_format_source() -> bool:
    return _FORMAT_SOURCE


def set_device_sync(flag: bool):
    global _DEVICE_SYNC
    _DEVICE_SYNC = flag


def get_device_sync() -> bool:
    return _DEVICE_SYNC


def is_gpu_backend() -> bool:
    return get_backend().endswith("cuda") or get_backend().endswith("gpu")


def is_gtc_backend() -> bool:
    return get_backend().startswith("gtc")


def get_backend_opts(func: Callable) -> Dict[str, Any]:
    backend_opts: Dict[str, Any] = {**_backend_options["all"]}

    stencil_name = f"{func.__module__.split('.')[-1]}.{func.__name__}"
    if stencil_name in _backend_options:
        backend_opts.update(_backend_options[stencil_name])

    if is_gpu_backend():
        backend_opts["device_sync"] = get_device_sync()

    return backend_opts


class StencilConfig(Hashable):
    def __init__(
        self,
        backend: str,
        rebuild: bool,
        validate_args: bool,
        format_source: bool,
        backend_opts: Dict[str, Any],
    ):
        self.backend = backend
        self.rebuild = rebuild
        self.validate_args = validate_args
        self.format_source = format_source
        self.backend_opts = backend_opts
        self._hash = self._compute_hash()

    def _compute_hash(self):
        md5 = hashlib.md5()
        md5.update(self.backend.encode())
        for attr in (
            self.rebuild,
            self.validate_args,
            self.format_source,
        ):
            md5.update(bytes(attr))
        return int(md5.hexdigest(), base=16)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        try:
            return self.__hash__() == other.__hash__()
        except AttributeError:
            return False

    @property
    def stencil_kwargs(self):
        kwargs = {
            "backend": self.backend,
            "rebuild": self.rebuild,
            "format_source": self.format_source,
        }
        if self.backend_opts:
            kwargs.update(self.backend_opts)
        return kwargs


def get_stencil_config(func: Callable):
    return StencilConfig(
        backend=get_backend(),
        rebuild=get_rebuild(),
        validate_args=get_validate_args(),
        format_source=get_format_source(),
        backend_opts=get_backend_opts(func),
    )


# Options: numpy, gtx86, gtcuda, debug
_BACKEND = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_FORMAT_SOURCE = getenv_bool("FV3_STENCIL_FORMAT_SOURCE", "False")
_VALIDATE_ARGS = True
_DEVICE_SYNC = False
