import functools
import hashlib
import os
from collections.abc import Hashable


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
@functools.lru_cache(maxsize=None)
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


@functools.lru_cache(maxsize=None)
def is_gpu_backend() -> bool:
    return get_backend().endswith("cuda") or get_backend().endswith("gpu")


@functools.lru_cache(maxsize=None)
def is_gtc_backend() -> bool:
    return get_backend().startswith("gtc") and not get_backend().endswith("numpy")


class StencilConfig(Hashable):
    def __init__(
        self,
        backend: str,
        rebuild: bool,
        validate_args: bool,
        format_source: bool,
        device_sync: bool,
    ):
        self.backend = backend
        self.rebuild = rebuild
        self.validate_args = validate_args
        self.format_source = format_source
        self.device_sync = device_sync
        self._hash = self._compute_hash()

    def _compute_hash(self):
        md5 = hashlib.md5()
        md5.update(self.backend.encode())
        for attr in (
            self.rebuild,
            self.validate_args,
            self.format_source,
            self.device_sync,
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
        if is_gpu_backend():
            kwargs["device_sync"] = self.device_sync
        return kwargs


def get_stencil_config():
    return StencilConfig(
        backend=get_backend(),
        rebuild=get_rebuild(),
        validate_args=get_validate_args(),
        format_source=get_format_source(),
        device_sync=get_device_sync(),
    )


# Options: numpy, gtx86, gtcuda, debug
_BACKEND = None
# If TRUE, all caches will bypassed and stencils recompiled
# if FALSE, caches will be checked and rebuild if code changes
_REBUILD = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "False")
_FORMAT_SOURCE = getenv_bool("FV3_STENCIL_FORMAT_SOURCE", "False")
_VALIDATE_ARGS = True
_DEVICE_SYNC = False
