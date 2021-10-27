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

    def __init__(self):
        self._inflight = {}
        self._times = {}
        self._names = {}
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
        json.dump(self._times, self._outfile_times, sort_keys=True, indent=4)
        json.dump(self._names, self._outfile_names, sort_keys=True, indent=4)
        self._outfile_times.close()
        self._outfile_names.close()

    def start_timestep(self):
        pass

    def end_timestep(self):
        pass


class NoneProfiler(BaseProfiler):
    """NoneProfiler default all operations to no-op"""

    def __init__(self):
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
                self._inflight_events[hash][key] = {}
            start_event = cp.cuda.Event()
            stop_event = cp.cuda.Event()
            self._inflight_events[hash][key]["start"] = start_event
            self._inflight_events[hash][key]["stop"] = stop_event
            start_event.record()
        else:
            raise NotImplementedError

    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        if profile_device == ProfileDevice.CPU:
            assert key in self._inflight[hash].keys()
            self.log(hash, key, time.perf_counter() - self._inflight[hash][key])
            self._inflight[hash][key] = None
        elif profile_device == ProfileDevice.HARDWARE:
            assert key in self._inflight_events[hash].keys()
            self._inflight_events[hash][key]["stop"].record()
        else:
            raise NotImplementedError

    def end_timestep(self):
        self._gather_device_times()
        return super().end_timestep()

    def _gather_device_times(self):
        cp.cuda.runtime.deviceSynchronize()
        for hash, stencils in self._inflight_events.items():
            for key, events in stencils.items():
                time_ms = cp.cuda.get_elapsed_time(events["start"], events["stop"])
                self.log(hash, key, time_ms / 1000)
                stencils = {}


class CPUProfiler(BaseProfiler):
    """TO BE IMPLEMENTED"""

    def __init__(self):
        super().__init__()

    def start(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        return NotImplementedError

    def stop(self, hash, key: str, profile_device=ProfileDevice.HARDWARE):
        return NotImplementedError
