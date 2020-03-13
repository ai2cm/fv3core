import pytest
import sys
import importlib
import warnings
import os
import fv3
import fv3._config
import fv3.utils.gt4py_utils
import fv3.translate
import collections

sys.path.append("/serialbox2/install/python")  # noqa
import serialbox


GRID_SAVEPOINT_NAME = "Grid-Info"
PARALLEL_SAVEPOINT_NAMES = ["HaloUpdate"]


class ReplaceRepr:

    def __init__(self, wrapped, new_repr):
        self._wrapped = wrapped
        self._repr = new_repr

    def __repr__(self):
        return self._repr

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    fv3.utils.gt4py_utils.backend = backend
    return backend


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_from_config(pytestconfig)


def data_path_from_config(config):
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(data_path, "input.nml")
    fv3._config.set_namelist(namelist_filename)
    return data_path


@pytest.fixture
def serializer(data_path, rank):
    return get_serializer(data_path, rank)


def get_serializer(data_path, rank):
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
    )


def is_input_name(savepoint_name):
    return savepoint_name[-3:] == "-In"


def to_output_name(savepoint_name):
    return savepoint_name[-3:] + "-Out"


def make_grid(grid_savepoint, serializer):
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = read_serialized_data(serializer, grid_savepoint, field)
    return fv3.translate.translate.TranslateGrid(grid_data).python_grid()


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def process_grid_savepoint(serializer, grid_savepoint):
    grid = make_grid(grid_savepoint, serializer)
    fv3._config.set_grid(grid)
    return grid


def get_test_class_instance(test_name, grid):
    try:
        instance = getattr(fv3.translate, f"Translate{test_name.replace('-', '_')}")(
            grid
        )
    except (AttributeError, ImportError):
        instance = None
    return instance


def get_sequential_savepoint_names(metafunc, data_path):
    only_names = metafunc.config.getoption("which_modules")
    if only_names is None:
        savepoint_names = set()
        serializer = get_serializer(data_path, rank=0)
        for savepoint in serializer.savepoint_list():
            if is_input_name(savepoint.name):
                savepoint_names.add(savepoint.name[:-3])
    else:
        savepoint_names = set(only_names.split(","))
        savepoint_names.discard("")
    skip_names = metafunc.config.getoption("skip_modules")
    if skip_names is not None:
        savepoint_names.difference_update(skip_names.split(","))
    savepoint_names.difference_update(PARALLEL_SAVEPOINT_NAMES)
    return savepoint_names


def get_parallel_savepoint_names(metafunc, data_path):
    only_names = metafunc.config.getoption("which_modules")
    if only_names is None:
        savepoint_names = set(PARALLEL_SAVEPOINT_NAMES)
    else:
        savepoint_names = set(only_names.split(",")).intersection(
            PARALLEL_SAVEPOINT_NAMES
        )
    skip_names = metafunc.config.getoption("skip_modules")
    if skip_names is not None:
        savepoint_names.difference_update(skip_names.split(","))
    return savepoint_names


SavepointCase = collections.namedtuple(
    "SavepointCase",
    [
        "test_name",
        "rank",
        "serializer",
        "input_savepoints",
        "output_savepoints",
        "grid",
    ],
)


def sequential_savepoint_cases(metafunc, data_path):
    layout = fv3._config.namelist["layout"]
    total_ranks = 6 * layout[0] * layout[1]
    savepoint_names = get_sequential_savepoint_names(metafunc, data_path)
    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
        grid = process_grid_savepoint(serializer, grid_savepoint)
        for test_name in sorted(list(savepoint_names)):
            input_savepoints = serializer.get_savepoint(f"{test_name}-In")
            output_savepoints = serializer.get_savepoint(f"{test_name}-Out")
            check_savepoint_counts(test_name, input_savepoints, output_savepoints)
            yield SavepointCase(
                test_name, rank, serializer, input_savepoints, output_savepoints, grid
            )


def check_savepoint_counts(test_name, input_savepoints, output_savepoints):
    if len(input_savepoints) != len(output_savepoints):
        warnings.warn(
            f"number of input and output savepoints not equal for {test_name}:"
            f" {len(input_savepoints)} in and {len(output_savepoints)} out"
        )
    elif len(input_savepoints) == 0:
        warnings.warn(f"no savepoints found for {test_name}")


def parallel_savepoint_cases(metafunc, data_path):
    layout = fv3._config.namelist["layout"]
    total_ranks = 6 * layout[0] * layout[1]
    grid_list = []
    for rank in reversed(range(total_ranks)):
        serializer = get_serializer(data_path, rank)
        grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
        grid_list.append(process_grid_savepoint(serializer, grid_savepoint))
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    for test_name in sorted(list(savepoint_names)):
        input_list = []
        output_list = []
        for rank in reversed(range(total_ranks)):
            serializer = get_serializer(data_path, rank)
            input_savepoints = serializer.get_savepoint(f"{test_name}-In")
            output_savepoints = serializer.get_savepoint(f"{test_name}-Out")
            check_savepoint_counts(test_name, input_savepoints, output_savepoints)
            input_list.append(input_savepoints)
            output_list.append(output_savepoints)
        yield SavepointCase(
            test_name, None, serializer, zip(input_list), zip(output_list), grid_list
        )


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ == "test_sequential_savepoint":
        generate_sequential_stencil_tests(metafunc)
    if metafunc.function.__name__ == "test_parallel_savepoint":
        generate_parallel_stencil_tests(metafunc)


def generate_sequential_stencil_tests(metafunc):
    arg_names = [
        "testobj",
        "test_name",
        "serializer",
        "savepoint_in",
        "savepoint_out",
        "rank",
        "grid",
    ]
    if all(name in metafunc.fixturenames for name in arg_names):
        data_path = data_path_from_config(metafunc.config)
        _generate_stencil_tests(
            metafunc,
            arg_names,
            sequential_savepoint_cases(metafunc, data_path),
            get_sequential_param,
        )


def generate_parallel_stencil_tests(metafunc):
    arg_names = [
        "testobj",
        "test_name",
        "serializer",
        "savepoint_in",
        "savepoint_out",
        "grid",
    ]
    if all(name in metafunc.fixturenames for name in arg_names):
        data_path = data_path_from_config(metafunc.config)
        _generate_stencil_tests(
            metafunc,
            arg_names,
            parallel_savepoint_cases(metafunc, data_path),
            get_parallel_param,
        )


def _generate_stencil_tests(metafunc, arg_names, savepoint_cases, get_param):
    if all(name in metafunc.fixturenames for name in arg_names):
        param_list = []
        for case in savepoint_cases:
            testobj = get_test_class_instance(case.test_name, case.grid)
            max_call_count = min(len(case.input_savepoints), len(case.output_savepoints))
            for i, (savepoint_in, savepoint_out) in enumerate(
                zip(case.input_savepoints, case.output_savepoints)
            ):
                param_list.append(
                    get_param(case, testobj, savepoint_in, savepoint_out, i, max_call_count)
                )
        metafunc.parametrize(", ".join(arg_names), param_list)


def get_parallel_param(case, testobj, savepoint_in, savepoint_out, call_count, max_call_count):
    return pytest.param(
        testobj,
        case.test_name,
        ReplaceRepr(case.serializer, f"<Serializer>"),
        savepoint_in,
        savepoint_out,
        case.grid,
        id=f"{case.test_name}-call_count={call_count}",
        marks=pytest.mark.dependency(
            name=f"{case.test_name}-{call_count}",
            depends=[
                f"{case.test_name}-{lower_count}"
                for lower_count in range(0, call_count)
            ]
        ),
    )


def get_sequential_param(case, testobj, savepoint_in, savepoint_out, call_count, max_call_count):
    return pytest.param(
        testobj,
        case.test_name,
        # serializer repr is very verbose, and not all that useful, so we hide it here
        ReplaceRepr(case.serializer, f"<Serializer for rank {case.rank}>"),
        savepoint_in,
        savepoint_out,
        case.rank,
        case.grid,
        id=f"{case.test_name}-rank={case.rank}-call_count={call_count}",
        marks=pytest.mark.dependency(
            name=f"{case.test_name}-{case.rank}-{call_count}",
            depends=[
                f"{case.test_name}-{lower_rank}-{count}"
                for lower_rank in range(0, case.rank)
                for count in range(0, max_call_count)
            ] + [
                f"{case.test_name}-{case.rank}-{lower_count}"
                for lower_count in range(0, call_count)
            ]
        ),
    )


def pytest_addoption(parser):
    parser.addoption("--which_modules", action="store", default="all")
    parser.addoption("--skip_modules", action="store", default="none")
    parser.addoption("--print_failures", action="store_true")
    parser.addoption("--failure_stride", action="store", default=1)
    parser.addoption("--data_path", action="store", default="./")
    parser.addoption("--backend", action="store", default="numpy")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "sequential(name): mark test as running sequentially on ranks"
    )
    config.addinivalue_line(
        "markers", "parallel(name): mark test as running in parallel across ranks"
    )
