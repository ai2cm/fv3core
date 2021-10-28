import time
import abc
from enum import Enum
from datetime import datetime
import json

from mpi4py import MPI

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import copy


class OrderedEnum(Enum):
    """As per Python documentation https://docs.python.org/3/library/enum.html#orderedenum"""

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class ProfileLevel(OrderedEnum):
    """Granularity of profiling.

    NONE: no profiling - defaulting to NoneProfiler
    TIMINGS: low impact, init & runtime time profiling
    ALL: _exec_info & everything else
    """

    NONE = 0
    TIMINGS = 1
    ALL = 2


class ProfileDevice(Enum):
    """Which target the profiler should look at.

    HARDWARE: look at own hardware specific timer
    CPU: read CPU timing
    """

    HARDWARE = 1
    CPU = 2


class BaseProfiler(abc.ABC):
    """Base profiler establishnig the API for all backend specific profilers"""

    _TIMESTEP_HASH: int = 0

    def __init__(self):
        self._timestep = 0
        self._inflight = {}
        self._times = {}
        self._names = {}
        self._timesteps = {"timestep": []}
        # Open the files here rather than in _dump to go around the order
        # of teardown error raised by using __del__
        self._filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_r{MPI.COMM_WORLD.Get_rank()}"
        self._outfile_times = open(f"{self._filename}_times.json", "w")
        self._outfile_names = open(f"{self._filename}_names.json", "w")

    def __del__(self):
        print("[PROFILER] Dumping in .json")
        self._dump()

    @abc.abstractmethod
    def start(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        raise NotImplementedError

    def add(self, hash, name: str) -> "BaseProfiler":
        """Add an entry in the timings"""
        self._times[hash] = {}
        self._inflight[hash] = {}
        self._names[hash] = name
        return self

    def log(self, hash, key: str, time: float) -> "BaseProfiler":
        """Log a timing"""
        if key not in self._times[hash]:
            self._times[hash][key] = []
        self._times[hash][key].append(time)
        return self

    def _dump(self):
        """Dump all profiled information in .json format"""
        json.dump(dict(self._timesteps), self._outfile_times, sort_keys=True, indent=4)
        json.dump(self._names, self._outfile_names, sort_keys=True, indent=4)
        self._outfile_times.close()
        self._outfile_names.close()

    def _clear_times(self):
        for hash, categories in self._times.items():
            for key, values in categories.items():
                values.clear()

    def start_timestep(self):
        assert self._inflight[self._TIMESTEP_HASH] is None
        self._inflight[self._TIMESTEP_HASH] = time.perf_counter()
        pass

    def end_timestep(self):
        assert self._inflight[self._TIMESTEP_HASH] is not None
        timestep_time = time.perf_counter() - self._inflight[self._TIMESTEP_HASH]

        self._timesteps["timestep"].append(
            {
                "t": self._timestep,
                "overall_time": timestep_time,
                "times": copy.deepcopy(self._times),
            }
        )

        self._clear_times()
        self._inflight[self._TIMESTEP_HASH] = None
        self._timestep += 1


class NoneProfiler(BaseProfiler):
    """NoneProfiler default all operations to no-op"""

    def __init__(self):
        pass

    def __del__(self):
        pass

    def add(self, hash, name: str) -> "BaseProfiler":
        return self

    def log(self, hash, key: str, time: float) -> "BaseProfiler":
        return self

    def start(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        pass

    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        pass


class CUDAProfiler(BaseProfiler):
    """CUDAProfiler capabale of CUDA specific profiling

    Using cudaEvent for kernel timing. All Events are processed after
    a global sync at the end of a timestep.
    """

    class CUDAEventTimer:
        def __init__(self):
            self._start_event = cp.cuda.Event()
            self._stop_event = cp.cuda.Event()

        def start(self):
            self._start_event.record()

        def stop(self):
            self._stop_event.record()

        def time(self) -> float:
            return cp.cuda.get_elapsed_time(self._start_event, self._stop_event)

    def __init__(self):
        super().__init__()
        self._inflight_events = {}

    def add(self, hash, name: str) -> "BaseProfiler":
        self._inflight_events[hash] = {}
        return super().add(hash, name)

    def start(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        if profile_device == ProfileDevice.CPU:
            self._inflight[hash][key] = time.perf_counter()
        elif profile_device == ProfileDevice.HARDWARE:
            if key not in self._inflight_events[hash].keys():
                self._inflight_events[hash][key] = []
            event_timer = CUDAProfiler.CUDAEventTimer()
            self._inflight_events[hash][key].append(event_timer)
            event_timer.start()
        else:
            raise NotImplementedError

    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        if profile_device == ProfileDevice.CPU:
            assert key in self._inflight[hash].keys()
            self.log(hash, key, time.perf_counter() - self._inflight[hash][key])
            self._inflight[hash][key] = None
        elif profile_device == ProfileDevice.HARDWARE:
            assert key in self._inflight_events[hash].keys()
            self._inflight_events[hash][key][-1].stop()
        else:
            raise NotImplementedError

    def end_timestep(self):
        self._gather_device_times()
        return super().end_timestep()

    def _gather_device_times(self):
        cp.cuda.runtime.deviceSynchronize()
        for hash, stencils in self._inflight_events.items():
            for key, events in stencils.items():
                for event in events:
                    self.log(hash, key, event.time() / 1000)
                events.clear()


class CPUProfiler(BaseProfiler):
    """TO BE IMPLEMENTED"""

    def __init__(self):
        super().__init__()

    def start(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        return NotImplementedError

    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        return NotImplementedError
