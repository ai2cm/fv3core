# -*- coding: utf-8 -*-
import abc
import datetime as dt
import numpy as np
import os
import random
import sqlite3
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Set, Union

from fv3core.utils.mpi import MPI

from gt4py.definitions import FieldInfo
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


class StencilPool(object, metaclass=Container):
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._futures: Dict[int, object] = {}

    def __call__(self, stencil_id: int, builder: "StencilBuilder"):
        if stencil_id not in self._futures:
            self._futures[stencil_id] = self._executor.submit(builder.backend.generate)
        return self._futures[stencil_id]

    def __contains__(self, stencil_id: int):
        return stencil_id in self._futures

    def __getitem__(self, stencil_id: int) -> int:
        return self._futures[stencil_id]

    def clear(self) -> None:
        self._futures.clear()


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


class SqliteTable(StencilTable):
    def __init__(self, db_file: str = "gt4py.db"):
        super().__init__()
        self._conn = sqlite3.connect(db_file)
        if self._conn:
            create_table_sql = """CREATE TABLE IF NOT EXISTS stencils(
                                    id integer PRIMARY KEY,
                                    stencil integer NOT NULL,
                                    node integer NOT NULL);"""
            cursor = self._conn.cursor()
            cursor.execute(create_table_sql)

    def __del__(self):
        if self._conn:
            self._conn.close()

    def __getitem__(self, key: int) -> int:
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT node FROM stencils WHERE stencil={key}")
        rows = cursor.fetchall()
        if rows:
            value = int(rows[0][0])
            if value == self.DONE_STATE:
                self._finished_keys.add(key)
            return value
        return self.NONE_STATE

    def __setitem__(self, key: int, value: int) -> None:
        sql = """INSERT INTO stencils(stencil, node) VALUES(?,?)"""
        cursor = self._conn.cursor()
        cursor.execute(sql, (key, value))
        self._conn.commit()


class WindowTable(StencilTable):
    MAX_STENCILS: int = 100

    def __init__(self, comm: Optional[Any] = None, max_size: int = 100):
        super().__init__()
        self._node_id = comm.Get_rank()
        self._n_nodes = comm.Get_size()
        self._key_nodes: Dict[int, int] = dict()

        int_size = MPI.INT.Get_size()
        self._buffer_size = max_size + 1
        self._window_size = int_size * self._buffer_size
        self._window = MPI.Win.Allocate(self._window_size, int_size, comm=comm)

        self._buffer: np.ndarray = np.empty(self._buffer_size, dtype=np.int32)
        self._initialize(self.NONE_STATE)
        comm.Barrier()

    def to_dict(self) -> Dict[int, Union[int, np.ndarray]]:
        table_dict = dict(node_id=self._node_id, n_nodes=self._n_nodes, buffers=[])
        for n in range(self._n_nodes):
            buffer = self._get_buffer(n)
            table_dict["buffers"].append(buffer)
        return table_dict

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
                n_stencils = buffer[0]
                for n in range(n_stencils):
                    index = n * 2 + 1
                    if buffer[index] == key:
                        value = buffer[index] + 1
                        self._key_nodes[key] = (node_id, index)
                        break

        if value == self.DONE_STATE:
            self._finished_keys.add(key)
        return value

    def __setitem__(self, key: int, value: int) -> None:
        if value == self.DONE_STATE:
            self._finished_keys.add(key)

        if key in self._key_nodes:
            index = self._key_nodes[key][1]
        else:
            n_stencils = self._buffer[0]
            index: int = -1
            for n in range(n_stencils):
                pos = n * 2 + 1
                if self._buffer[pos] == key:
                    index = pos
                    break
            # New entry...
            if index < 0:
                index = n_stencils * 2 + 1
                self._buffer[0] = n_stencils + 1

        self._buffer[index] = key
        self._buffer[index + 1] = value
        self._set_buffer()
        self._key_nodes[key] = (self._node_id, index)

    def _initialize(self, value: int):
        self._buffer.fill(value)
        self._buffer[0] = 0
        self._set_buffer()

    def _set_buffer(self):
        # with open(f"./caching_r{self._node_id}.log", "a") as log:
        #     log.write(f"{dt.datetime.now()}: R{self._node_id}: W: {value}\n")
        self._window.Lock(self._node_id)
        self._window.Put([self._buffer, MPI.INT], self._node_id)
        self._window.Unlock(self._node_id)

    def _get_buffer(self, node_id: int) -> np.ndarray:
        if node_id != self._node_id:
            buffer = np.empty(self._buffer_size, dtype=np.int32)
            self._window.Lock(node_id)
            self._window.Get([buffer, MPI.INT], node_id)
            self._window.Unlock(node_id)
            # with open(f"./caching_r{self._node_id}.log", "a") as log:
            #     log.write(f"{dt.datetime.now()}: R{self._node_id}: R: {buffer[0]} from {node_id}\n")
            return buffer
        return self._buffer


class FutureStencil:
    """
    A stencil object that is compiled by another node in a distributed context.
    """

    _thread_pool: StencilPool = StencilPool()
    _id_table: StencilTable = RedisTable()
    # _id_table: StencilTable = SqliteTable()
    # _id_table: StencilTable = WindowTable()

    def __init__(self, builder: Optional["StencilBuilder"] = None):
        self._builder: Optional["StencilBuilder"] = builder
        self._stencil_object: Optional[StencilObject] = None
        self._sleep_time: float = 0.3
        self._timeout: float = 60.0

    @property
    def is_built(self) -> bool:
        return self._stencil_object is not None

    @property
    def stencil_object(self) -> StencilObject:
        if self._stencil_object is None:
            self.wait_for_stencil()
        return self._stencil_object

    @property
    def field_info(self) -> Dict[str, FieldInfo]:
        return self.stencil_object.field_info

    def delay(self, factor: float = 1.0, use_random: bool = False) -> float:
        delay_time = (random.random() if use_random else self._sleep_time) * factor
        time.sleep(delay_time)
        return delay_time

    def wait_for_stencil(self):
        builder = self._builder
        cache_info_path = builder.caching.cache_info_path
        node_id = MPI.COMM_WORLD.Get_rank() if MPI else 0
        stencil_id = int(builder.stencil_id.version, 16)
        stencil_class = None if builder.options.rebuild else builder.backend.load()

        if not stencil_class:
            # Random delay before accessing distributed dict...
            self.delay(0.25, True)
            if self._id_table.is_none(stencil_id):
                # Stencil not yet compiled or in progress so claim it...
                self._id_table[stencil_id] = node_id
                # stencil_class = builder.backend.generate()
                self._thread_pool(stencil_id, builder)
                with open(f"./caching_r{node_id}.log", "a") as log:
                    log.write(
                        f"{dt.datetime.now()}: R{node_id}: Submitted stencil '{cache_info_path.stem}' ({stencil_id})\n"
                    )
            else:
                if not self._id_table.is_done(stencil_id):
                    # Wait for stencil to be done...
                    with open(f"./caching_r{node_id}.log", "a") as log:
                        log.write(
                            f"{dt.datetime.now()}: R{node_id}: Waiting for stencil '{cache_info_path.stem}' ({stencil_id})\n"
                        )
                    time_elapsed: float = 0.0
                    while not self._id_table.is_done(stencil_id) and time_elapsed < self._timeout:
                        time_elapsed += self.delay()
                    if time_elapsed >= self._timeout:
                        error_message = f"Timeout while waiting for stencil '{cache_info_path.stem}' to compile on R{node_id}"
                        with open(f"./caching_r{node_id}.log", "a") as log:
                            log.write(
                                f"{dt.datetime.now()}: R{node_id}: Timeout while waiting for stencil '{cache_info_path.stem}'\n"
                            )
                        raise RuntimeError(error_message)
                    # Wait a bit before loading...
                    self.delay(5.0)

                with open(f"./caching_r{node_id}.log", "a") as log:
                    log.write(
                        f"{dt.datetime.now()}: R{node_id}: Loading stencil '{cache_info_path.stem}' ({stencil_id})\n"
                    )
                stencil_class = builder.backend.load()

            if stencil_id in self._thread_pool:
                future = self._thread_pool[stencil_id]
                stencil_class = future.result()
                # Set to DONE...
                self._id_table.set_done(stencil_id)

        if not stencil_class:
            error_message = f"`stencil_class` is None '{cache_info_path.stem}' ({stencil_id})!"
            with open(f"./caching_r{node_id}.log", "a") as log:
                log.write(f"{dt.datetime.now()}: R{node_id}: ERROR: {error_message}\n")
                raise RuntimeError(error_message)

        with open(f"./caching_r{node_id}.log", "a") as log:
            log.write(
                f"{dt.datetime.now()}: R{node_id}: Finished stencil '{cache_info_path.stem}' ({stencil_id})\n"
            )
        self._stencil_object = stencil_class()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # If args or kwargs supplied, call stencil object, instantiate otherwise
        if args or kwargs:
            return (self.stencil_object)(*args, **kwargs)
        return self

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.stencil_object.run(*args, **kwargs)
