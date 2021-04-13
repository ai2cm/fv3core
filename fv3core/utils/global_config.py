import os


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


def set_do_halo_exchange(flag: bool):
    global _DO_HALO_EXCHANGE
    _DO_HALO_EXCHANGE = flag


def get_do_halo_exchange() -> bool:
    return _DO_HALO_EXCHANGE


def set_device_sync(flag: bool):
    global _DEVICE_SYNC
    _DEVICE_SYNC = flag


def get_device_sync() -> bool:
    return _DEVICE_SYNC


_BACKEND = None  # Options: numpy, gtx86, gtcuda, debug
_REBUILD = getenv_bool("FV3_STENCIL_REBUILD_FLAG", "True")
_FORMAT_SOURCE = getenv_bool("FV3_STENCIL_FORMAT_SOURCE", "False")
_DO_HALO_EXCHANGE = True
_VALIDATE_ARGS = True
_DEVICE_SYNC = False
