import types
from typing import List

import gt4py


PYTHON_UNSAFE_STORAGES: List[gt4py.storage.storage.Storage] = []


def _unsafe_storage_access(storages):
    for s1 in storages:
        for s2 in PYTHON_UNSAFE_STORAGES:
            if s1 is s2:
                return True
    return False


def requires_safe(func, method_parent=None):
    def wrapped(*args, **kwargs):
        if __debug__ and _unsafe_storage_access(list(args) + list(kwargs.values())):
            raise ValueError("operation requires you call utils.device_sync() first")
        return func(*args, **kwargs)

    if method_parent is not None:
        wrapped = types.MethodType(wrapped, method_parent)
    return wrapped


for storage_type in (
    gt4py.storage.storage.CPUStorage,
    gt4py.storage.storage.GPUStorage,
):
    storage_type.__getitem__ = requires_safe(storage_type.__getitem__)
