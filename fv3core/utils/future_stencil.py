# -*- coding: utf-8 -*-
import abc
import datetime as dt
import numpy as np
import os
import random
import sqlite3
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Set

from fv3core.utils.mpi import MPI

from gt4py.definitions import FieldInfo
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject


try:
    from redis_dict import RedisDict
except ModuleNotFoundError:
    redis_dict = None


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StencilPool(object, metaclass=Singleton):
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


class StencilTable(object, metaclass=Singleton):
    DONE_STATE: int = -1
    NONE_STATE: int = -2

    def __init__(self):
        self._finished_keys: Set[int] = set()

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
    def __init__(self, comm: Optional[Any] = None):
        super().__init__()
        self._node_id = comm.Get_rank()
        self._n_nodes = comm.Get_size()
        self._key_nodes: Dict[int, int] = dict()
        disp_unit = MPI.INT.Get_size()
        self._window = MPI.Win.Allocate(disp_unit, disp_unit, comm=comm)
        self._set_value(self.NONE_STATE)
        comm.Barrier()

    def __getitem__(self, key: int) -> int:
        if key in self._finished_keys:
            return self.DONE_STATE

        value: int = self.NONE_STATE
        if key in self._key_nodes:
            node_id = self._key_nodes[key]
            value = self._get_value(node_id)
        else:
            for node_id in range(self._n_nodes):
                if node_id != self._node_id:
                    value = self._get_value(node_id)
                    if value == key:
                        self._key_nodes[key] = node_id
                        break

        if value == self.DONE_STATE:
            self._finished_keys.add(key)
        return value

    def __setitem__(self, key: int, value: int) -> None:
        if value == self.DONE_STATE:
            self._finished_keys.add(key)
            self._set_value(value)
        else:
            self._set_value(key)

    def _set_value(self, value: int):
        buffer = np.zeros(1, np.int)
        buffer[0] = value
        with open(f"./caching_r{self._node_id}.log", "a") as log:
            log.write(f"{dt.datetime.now()}: R{self._node_id}: W: {value}\n")
        self._window.Lock(self._node_id)
        self._window.Put([buffer, MPI.INT], self._node_id)
        self._window.Unlock(self._node_id)

    def _get_value(self, node_id: int) -> int:
        buffer = np.zeros(1, np.int)
        self._window.Lock(node_id)
        self._window.Get([buffer, MPI.INT], node_id)
        self._window.Unlock(node_id)
        with open(f"./caching_r{self._node_id}.log", "a") as log:
            log.write(f"{dt.datetime.now()}: R{self._node_id}: R: {buffer[0]} from {node_id}\n")
        return buffer[0]


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
