# -*- coding: utf-8 -*-
import abc
import datetime as dt
import numpy as np
import random
import time

from typing import Any, Callable, Dict, Optional, Set, Tuple

from fv3core.utils.mpi import MPI

from gt4py.definitions import BuildOptions, FieldInfo
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject

try:
    from redis_dict import RedisDict
except ModuleNotFoundError:
    redis_dict = None


class Container(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Container, cls).__call__(*args, **kwargs)
            cls._instances[cls].clear()
        return cls._instances[cls]


class StencilTable(object, metaclass=Container):
    DONE_STATE: int = -1
    NONE_STATE: int = -2

    def __init__(self):
        self._finished_keys: Set[int] = set()

    def clear(self) -> None:
        self._finished_keys.clear()

    def set_done(self, key: int) -> None:
        self[key] = self.DONE_STATE
        self._finished_keys.add(key)

    def is_done(self, key: int) -> bool:
        if key in self._finished_keys:
            return True
        if self[key] == self.DONE_STATE:
            self._finished_keys.add(key)
            return True
        return False

    def is_none(self, key: int) -> bool:
        return self[key] == self.NONE_STATE

    @abc.abstractmethod
    def __getitem__(self, item: int) -> int:
        pass

    @abc.abstractmethod
    def __setitem__(self, key: int, value: int) -> None:
        pass


class RedisTable(StencilTable):
    def __init__(self):
        super().__init__()
        self._dict: Dict[int, int] = RedisDict(namespace="gt4py")

    def clear(self) -> None:
        super().clear()
        self._dict.clear()

    def __getitem__(self, key: int) -> int:
        if key in self._dict:
            value = int(self._dict[key])
            if value == self.DONE_STATE:
                self._finished_keys.add(key)
            return value
        return self.NONE_STATE

    def __setitem__(self, key: int, value: int) -> None:
        self._dict[key] = value


class WindowTable(StencilTable):
    def __init__(self, comm: Optional[Any] = None, max_size: int = 200):
        super().__init__()
        if not comm:
            comm = MPI.COMM_WORLD
        self._node_id = comm.Get_rank()
        self._n_nodes = comm.Get_size()
        self._key_nodes: Dict[int, int] = dict()

        self._buffer_size = 2 * max_size + 1
        self._comm = comm
        self._initialize(self.NONE_STATE)

    def clear(self) -> None:
        super().clear()
        self.__init__()

    def __getitem__(self, key: int) -> int:
        if key in self._finished_keys:
            return self.DONE_STATE

        value: int = self.NONE_STATE
        if key in self._key_nodes:
            node_id, index = self._key_nodes[key]
            buffer = self._get_buffer(node_id)
            assert buffer[index] == key
            value = buffer[index + 1]
        else:
            for node_id in range(self._n_nodes):
                buffer = self._get_buffer(node_id)
                n_items = buffer[0]
                for n in range(n_items):
                    index = n * 2 + 1
                    if buffer[index] == key:
                        value = buffer[index + 1]
                        self._key_nodes[key] = (node_id, index)
                        break

        if value == self.DONE_STATE:
            self._finished_keys.add(key)

        return value

    def __setitem__(self, key: int, value: int) -> None:
        if value == self.DONE_STATE:
            self._finished_keys.add(key)

        buffer = self._get_buffer()
        if key in self._key_nodes:
            index = self._key_nodes[key][1]
        else:
            n_items = buffer[0]
            index: int = -1
            for n in range(n_items):
                pos = n * 2 + 1
                if buffer[pos] == key:
                    index = pos
                    break
            # New entry...
            if index < 0:
                index = n_items * 2 + 1
                buffer[0] = n_items + 1

        buffer[index] = key
        buffer[index + 1] = value
        self._set_buffer(buffer)
        self._key_nodes[key] = (self._node_id, index)

    def _initialize(self, value: int):
        self._mpi_type = MPI.LONG
        int_size = self._mpi_type.Get_size()
        self._np_type = np.int64
        self._window_size = (
            int_size * self._buffer_size * self._n_nodes if self._node_id == 0 else 0
        )
        self._window = MPI.Win.Allocate(
            size=self._window_size, disp_unit=int_size, comm=self._comm
        )

        if self._node_id == 0:
            buffer = np.frombuffer(self._window, dtype=self._np_type)
            buffer[:] = np.full(len(buffer), value, dtype=self._np_type)
            for n in range(self._n_nodes):
                buffer[n * self._buffer_size] = 0

        self._comm.Barrier()

    def _get_target(self, node_id: int = -1) -> Tuple[int]:
        if node_id < 0:
            node_id = self._node_id
        return (node_id * self._buffer_size, self._buffer_size, self._mpi_type)

    def _set_buffer(self, buffer: np.ndarray):
        self._window.Lock(rank=0)
        self._window.Put(buffer, target_rank=0, target=self._get_target())
        self._window.Unlock(rank=0)

    def _get_buffer(self, node_id: int = -1) -> np.ndarray:
        buffer = np.empty(self._buffer_size, dtype=self._np_type)
        self._window.Lock(rank=0)
        self._window.Get(buffer, target_rank=0, target=self._get_target(node_id))
        self._window.Unlock(rank=0)

        return buffer


def future_stencil(
    backend: Optional[str] = None,
    definition: Optional[Callable] = None,
    *,
    externals: Optional[Dict[str, Any]] = None,
    rebuild: bool = False,
    **kwargs: Any,
):
    """
    Create a future stencil object with deferred building in a distributed context

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`FutureStencil`
            Wrapper around an instance of the dynamically-generated subclass of :class:`gt4py.StencilObject`.
            Defers the generation step until the last moment and allows syntax checking independently.
            Also gives access to a more fine grained generate / build process.
    """
    def _decorator(func):
        options = BuildOptions(
            **{
                **StencilBuilder.default_options_dict(func),
                **StencilBuilder.name_to_options_args(""),
                "rebuild": rebuild,
                "build_info": {},
                **StencilBuilder.nest_impl_options(kwargs),
            }
        )
        stencil = FutureStencil(
            StencilBuilder(func, backend=backend, options=options).with_externals(externals or {})
        )
        return stencil

    if definition is None:
        return _decorator
    return _decorator(definition)


class FutureStencil:
    """
    A stencil object that is compiled by another node in a distributed context.
    """

    # _id_table: StencilTable = RedisTable()
    _id_table: StencilTable = WindowTable()

    def __init__(self, builder: Optional["StencilBuilder"] = None):
        self._builder: Optional["StencilBuilder"] = builder
        self._stencil_object: Optional[StencilObject] = None
        self._sleep_time: float = 0.3
        self._timeout: float = 60.0

    @classmethod
    def clear(cls):
        cls._id_table.clear()

    @property
    def cache_info_path(self) -> str:
        return self._builder.caching.cache_info_path.stem

    @property
    def stencil_object(self) -> StencilObject:
        if self._stencil_object is None:
            self._wait_for_stencil()
        return self._stencil_object

    @property
    def field_info(self) -> Dict[str, FieldInfo]:
        return self.stencil_object.field_info

    def _delay(self, factor: float = 1.0, use_random: bool = False) -> float:
        delay_time = (random.random() if use_random else self._sleep_time) * factor
        time.sleep(delay_time)
        return delay_time

    def _compile_stencil(self, node_id: int, stencil_id: int) -> Callable:
        # Stencil not yet compiled or in progress so claim it...
        self._id_table[stencil_id] = node_id

        with open(f"./caching_r{node_id}.log", "a") as log:
            log.write(
                f"{dt.datetime.now()}: R{node_id}: Compiling stencil '{self.cache_info_path}' ({stencil_id})\n"
            )

        stencil_class = self._builder.backend.generate()
        self._id_table.set_done(stencil_id)  # Set to DONE...

        return stencil_class

    def _load_stencil(self, node_id: int, stencil_id: int) -> Callable:
        if not self._id_table.is_done(stencil_id):
            # Wait for stencil to be done...
            with open(f"./caching_r{node_id}.log", "a") as log:
                log.write(
                    f"{dt.datetime.now()}: R{node_id}: Waiting for stencil '{self.cache_info_path}' ({stencil_id})\n"
                )

            time_elapsed: float = 0.0
            while (
                not self._id_table.is_done(stencil_id) and time_elapsed < self._timeout
            ):
                time_elapsed += self._delay()

            if time_elapsed >= self._timeout:
                error_message = f"Timeout while waiting for stencil '{self.cache_info_path}' to compile on R{node_id}"
                with open(f"./caching_r{node_id}.log", "a") as log:
                    log.write(
                        f"{dt.datetime.now()}: R{node_id}: Timeout while waiting for stencil '{self.cache_info_path}'\n"
                    )
                raise RuntimeError(error_message)
            # Wait a bit before loading...
            self._delay(5.0)

        with open(f"./caching_r{node_id}.log", "a") as log:
            log.write(
                f"{dt.datetime.now()}: R{node_id}: Loading stencil '{self.cache_info_path}' ({stencil_id})\n"
            )
        stencil_class = self._builder.backend.load()

        return stencil_class

    def _wait_for_stencil(self):
        builder = self._builder
        node_id = MPI.COMM_WORLD.Get_rank() if MPI else 0
        stencil_id = int(builder.stencil_id.version, 16)
        stencil_class = None if builder.options.rebuild else builder.backend.load()

        if not stencil_class:
            # Random delay before accessing distributed dict...
            self._delay(0.25, True)
            if self._id_table.is_none(stencil_id):
                stencil_class = self._compile_stencil(node_id, stencil_id)
            else:
                stencil_class = self._load_stencil(node_id, stencil_id)

        if not stencil_class:
            error_message = (
                f"`stencil_class` is None '{self.cache_info_path}' ({stencil_id})!"
            )
            with open(f"./caching_r{node_id}.log", "a") as log:
                log.write(f"{dt.datetime.now()}: R{node_id}: ERROR: {error_message}\n")
                raise RuntimeError(error_message)

        with open(f"./caching_r{node_id}.log", "a") as log:
            log.write(
                f"{dt.datetime.now()}: R{node_id}: Finished stencil '{self.cache_info_path}' ({stencil_id})\n"
            )

        self._stencil_object = stencil_class()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return (self.stencil_object)(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.stencil_object.run(*args, **kwargs)
