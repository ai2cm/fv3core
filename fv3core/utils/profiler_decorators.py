from fv3core.utils.global_config import get_profiler


def do_profile_timestep(func):
    def wrapper(*args, **kwargs):
        get_profiler().start_timestep()
        func(*args, **kwargs)
        get_profiler().end_timestep()

    return wrapper


def do_profile_object_hash(key):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            get_profiler().start(self._hash, key)
            func(self, *args, **kwargs)
            get_profiler().stop(self._hash, key)

        return wrapper

    return decorator
