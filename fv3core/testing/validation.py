import inspect
from typing import Tuple

import numpy as np

import fv3core.stencils.updatedzd


def get_selective_class(
    cls,
    name_to_function_dict,
):
    """
    Convert a model class into one that sets nans on non-validated outputs,
    and gives a helper function to retrieve the output subset we want to validate.

    Using this ensures that if these non-validated values are ever used, a test
    will fail because an output will have NaN.
    """

    class SelectivelyValidated:
        """
        Wrapper class that sets non-validated outputs to nan, and gives a helper
        function to retrieve the output subset to be validated.
        """

        def __init__(self, *args, **kwargs):

            selective_arg_names = []
            origin_domain_funcs = []

            for argument_name in name_to_function_dict.keys():
                selective_arg_names.append(argument_name)
                selective_arg_names.append(
                    name_to_function_dict[argument_name]["savepoint_name"]
                )
                origin_domain_funcs.append(
                    name_to_function_dict[argument_name]["origin_domain_func"]
                )
                origin_domain_funcs.append(
                    name_to_function_dict[argument_name]["origin_domain_func"]
                )

            self.wrapped = cls(*args, **kwargs)
            origin = []
            domain = []
            for variable_origin, variable_domain in [
                origin_domain_func(self.wrapped)
                for origin_domain_func in origin_domain_funcs
            ]:
                origin.append(variable_origin)
                domain.append(variable_domain)

            self._validation_slice = {}
            for i in range(len(selective_arg_names)):
                self._validation_slice[selective_arg_names[i]] = tuple(
                    slice(start, start + n) for start, n in zip(origin[i], domain[i])
                )

            self._all_argument_names = tuple(
                inspect.getfullargspec(self.wrapped).args[1:]
            )
            assert "self" not in self._all_argument_names
            self._selective_argument_names = selective_arg_names

        def __call__(self, *args, **kwargs):
            kwargs.update(self._args_to_kwargs(args))
            self.wrapped(**kwargs)
            self._set_nans(kwargs)

        def _args_to_kwargs(self, args):
            return dict(zip(self._all_argument_names, args))

        def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
            """
            Given an output array, return the slice of the array which we'd
            like to validate against reference data
            """
            if varname in self._selective_argument_names:
                output = output[self._validation_slice[varname]]
            return output

        def _set_nans(self, kwargs):
            for name in set(kwargs.keys()).intersection(self._selective_argument_names):
                array = kwargs[name]
                validation_data = np.copy(array[self._validation_slice[name]])
                array[:] = np.nan
                array[self._validation_slice[name]] = validation_data

    return SelectivelyValidated


def get_update_height_on_d_grid_selective_domain(
    instance,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    origin = instance.grid.compute_origin()
    domain = instance.grid.domain_shape_compute(add=(0, 0, 1))
    return origin, domain


def get_update_height_on_c_grid_selective_domain_2d(
    instance,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    origin = (instance.grid.is_, instance.grid.js)
    domain = (instance.grid.npx, instance.grid.npy)
    return origin, domain


def get_update_height_on_c_grid_selective_domain_3d(
    instance,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    origin = instance.grid.compute_origin()
    domain = instance.grid.domain_shape_compute(add=(0, 0, 1))
    return origin, domain


def enable_selective_validation():
    """
    Replaces certain function-classes with wrapped versions that set data we aren't
    validating to NaN, and have an attribute function `subset_output` that
    takes in a string variable name and an output array and returns the
    subset of that array which should be validated.

    This wrapping removes any attributes of the wrapped module.
    """
    # to enable selective validation for a new class, add a new monkeypatch
    # this should require only a new function for (origin, domain)
    # note we have not implemented disabling selective validation once enabled
    fv3core.stencils.updatedzd.UpdateHeightOnDGrid = get_selective_class(
        fv3core.stencils.updatedzd.UpdateHeightOnDGrid,
        {
            "height": {
                "savepoint_name": "zh",
                "origin_domain_func": get_update_height_on_d_grid_selective_domain,
            }
        },  # must include both function and savepoint names
    )
    # make absolutely sure you don't write just the savepoint name, this would
    # selecively validate without making sure it's safe to do so

    # to enable selective validation for a new class, add a new monkeypatch
    # this should require only a new function for (origin, domain)
    # note we have not implemented disabling selective validation once enabled
    fv3core.stencils.updatedzc.UpdateGeopotentialHeightOnCGrid = get_selective_class(
        fv3core.stencils.updatedzc.UpdateGeopotentialHeightOnCGrid,
        {
            "ws": {
                "savepoint_name": "ws",
                "origin_domain_func": get_update_height_on_c_grid_selective_domain_2d,
            },
            "gz": {
                "savepoint_name": "gz",
                "origin_domain_func": get_update_height_on_c_grid_selective_domain_3d,
            },
        },  # must include both function and savepoint names
    )
    # make absolutely sure you don't write just the savepoint name, this would
    # selecively validate without making sure it's safe to do so
