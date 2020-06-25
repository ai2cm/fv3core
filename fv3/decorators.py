import collections
import functools
import types

ArgSpec = collections.namedtuple("ArgSpec", ["arg_name", "standard_name", "units", "intent"])
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs):
    for spec in arg_specs:
        if spec.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {spec.arg_name} is {spec.intent}, must be one of {VALID_INTENTS}"
            )
    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            kwargs = {}
            for spec in arg_specs:
                arg_name, standard_name, units, intent = spec
                if standard_name not in state:
                    raise ValueError(f"{standard_name} not present in state")
                elif units != state[standard_name].units:
                    raise ValueError(f"{standard_name} has units {state[standard_name].units} when {units} is required")
                else:
                    kwargs[spec.arg_name] = state[standard_name]
            return func(types.SimpleNamespace(**kwargs), *args, **kwargs)
        return func
    return decorator
