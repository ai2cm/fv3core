import pytest
import sys
import importlib
import warnings
import os
import fv3
import fv3._config
import fv3.utils.gt4py_utils
import fv3.translate.translate
import collections

sys.path.append("/serialbox2/install/python")  # noqa
import serialbox


GRID_SAVEPOINT_NAME = 'Grid-Info'


@pytest.fixture()
def backend(pytestconfig):
    return pytestconfig.getoption("backend")


@pytest.fixture()
def data_backend(pytestconfig, backend):
    data_backend = pytestconfig.getoption("data_backend")
    if data_backend is None:
        data_backend = backend
    fv3.utils.gt4py_utils.data_backend = data_backend


@pytest.fixture()
def exec_backend(pytestconfig, backend):
    exec_backend = pytestconfig.getoption("exec_backend")
    if exec_backend is None:
        exec_backend = backend
    fv3.utils.gt4py_utils.exec_backend = exec_backend


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_from_config(pytestconfig)


def data_path_from_config(config):
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(
        data_path, "input.nml"
    )
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
        module_name = f'fv3.translate.translate_{test_name.lower()}'
        module = importlib.import_module(module_name)
        instance = getattr(module, f"Translate{test_name.replace('-', '_')}")(grid)
    except (AttributeError, ImportError):
        instance = None
    return instance


def get_savepoint_names(metafunc, data_path, total_ranks):
    skip_names = metafunc.config.getoption("skip_modules").split(",")
    only_names = metafunc.config.getoption("which_modules").split(",")
    if '' in skip_names:
        skip_names.remove('')
    if '' in only_names:
        only_names.remove('')
    savepoint_names = set()
    for rank in range(total_ranks):
        serializer = get_serializer(data_path, rank)
        for savepoint in serializer.savepoint_list():
            if is_input_name(savepoint.name):
                savepoint_names.add(savepoint.name[:-3])
    savepoint_names.difference_update(skip_names)
    if len(only_names) > 0:
        savepoint_names.intersection_update(only_names)
    return savepoint_names


SavepointCase = collections.namedtuple(
    "SavepointCase",
    ["test_name", "rank", "serializer", "input_savepoints", "output_savepoints"]
)


def savepoint_cases(metafunc, data_path):
    layout = fv3._config.namelist['layout']
    total_ranks = 6 * layout[0] * layout[1]
    savepoint_names = get_savepoint_names(metafunc, data_path, total_ranks)
    for rank in reversed(range(total_ranks)):
        serializer = get_serializer(data_path, rank)
        for test_name in sorted(list(savepoint_names)):
            input_savepoints = serializer.get_savepoint(f'{test_name}-In')
            output_savepoints = serializer.get_savepoint(f'{test_name}-Out')
            if len(input_savepoints) != len(output_savepoints):
                warnings.warn(
                    f'number of input and output savepoints not equal for {test_name}:'
                    f' {len(input_savepoints)} in and {len(output_savepoints)} out')
            yield SavepointCase(
                test_name, rank, serializer, input_savepoints, output_savepoints
            )


def pytest_generate_tests(metafunc):
    generate_basic_stencil_tests(metafunc)


def generate_basic_stencil_tests(metafunc):
    arg_names = ["testobj", "test_name", "serializer", "savepoint_in", "savepoint_out", "rank", "grid"]
    if all(name in metafunc.fixturenames for name in arg_names):
        data_path = data_path_from_config(metafunc.config)
        serializer = get_serializer(data_path, rank=0)
        grid_savepoint = serializer.get_savepoint(GRID_SAVEPOINT_NAME)[0]
        grid = process_grid_savepoint(serializer, grid_savepoint)
        param_list = []

        for case in savepoint_cases(metafunc, data_path):
            testobj = get_test_class_instance(case.test_name, grid)
            for i, (savepoint_in, savepoint_out) in enumerate(zip(case.input_savepoints, case.output_savepoints)):
                param_list.append(
                    pytest.param(
                        testobj, case.test_name, case.serializer, savepoint_in, savepoint_out, case.rank, grid,
                        id=f"{case.test_name}-rank={case.rank}-call_count={i}"
                    )
                )
        arg_names = ["testobj", "test_name", "serializer", "savepoint_in", "savepoint_out", "rank", "grid"]
        # param_list = []
        metafunc.parametrize(", ".join(arg_names), param_list)


def pytest_addoption(parser):
    parser.addoption("--which_modules", action="store", default="")
    parser.addoption("--skip_modules", action="store", default="")
    parser.addoption("--data_path", action="store", default=".")
    parser.addoption("--data_backend", action="store", default=None)
    parser.addoption("--exec_backend", action="store", default=None)
    parser.addoption("--backend", action="store", default="numpy")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "basic(name): mark test as a basic test"
    )
    config.addinivalue_line(
        "markers", "halo(name): mark test as involving a halo update"
    )
