#!/usr/bin/env python3

import sys

sys.path.append("/serialbox2/install/python")  # noqa

import numpy as np
import serialbox as ser
import fv3._config
import pytest


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def collect_input_data(testobj, serializer, savepoint,):
    input_data = {}
    for varname in (
        testobj.serialnames(testobj.in_vars["data_vars"]) + testobj.in_vars["parameters"]
    ):
        input_data[varname] = read_serialized_data(serializer, savepoint, varname)
    return input_data


def test_savepoint(
    testobj,
    test_name,
    grid,
    serializer,
    savepoint_in,
    savepoint_out,
    rank,
    exec_backend,
    data_backend,
    subtests,
):
    if testobj is None:
        pytest.skip(f'no translate object available for savepoint {test_name}')
    fv3._config.set_grid(grid)
    input_data = collect_input_data(testobj, serializer, savepoint_in)
    # run python version of functionality
    output = testobj.compute(input_data)
    for varname in testobj.serialnames(testobj.out_vars):
        ref_data = read_serialized_data(serializer, savepoint_out, varname)
        with subtests.test(varname):
            np.testing.assert_allclose(output[varname], ref_data, rtol=testobj.max_error)


def get_serializer(data_path, rank):
    return ser.Serializer(
        ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
    )


# def state_from_savepoint(serializer, savepoint, name_to_std_name):
#     properties = fv3util.fortran_info.properties_by_std_name
#     state = {}
#     for name, std_name in name_to_std_name.items():
#         array = serializer.read(name, savepoint)
#         extent = list(np.asarray(array.shape) - 2 * np.asarray(utils.origin))
#         state['air_temperature'] = fv3util.Quantity(
#             array,
#             dims=properties['air_temperature']['dims'],
#             units=properties['air_temperature']['units'],
#             origin=utils.origin,
#             extent=extent
#         )
#     return state


# def get_communicator(comm, layout):
#     partitioner = fv3util.CubedSpherePartitioner(
#         fv3util.TilePartitioner(layout)
#     )
#     communicator = fv3util.CubedSphereCommunicator(
#         comm, partitioner
#     )
#     return communicator


# def test_halo_update(data_path, subtests):
#     properties = fv3util.fortran_info.properties_by_std_name
#     total_ranks = 6
#     layout = (1, 1)
#     shared_buffer = {}
#     states = []
#     communicators = []
#     for rank in range(total_ranks):
#         serializer = get_serializer(data_path, rank)
#         savepoint = serializer.savepoint['HaloUpdate-In']
#         state = state_from_savepoint(serializer, savepoint, {"array": "air_temperature"})
#         states.append(state)
#         comm = fv3util.testing.DummyComm(rank, total_ranks, buffer_dict=shared_buffer)
#         communicator = get_communicator(comm, layout)
#         communicator.start_halo_update(state['air_temperature'], n_ghost=utils.n_ghost)
#         communicators.append(communicator)
#     for rank, (state, communicator) in enumerate(zip(states, communicators)):
#         serializer = ser.Serializer(
#             ser.OpenModeKind.Read, data_path, "Generator_rank" + str(rank)
#         )
#         savepoint = serializer.savepoint['HaloUpdate-Out']
#         array = serializer.read("array", savepoint)
#         quantity = states['air_temperature']
#         communicator.finish_halo_update(quantity, n_ghost=utils.n_ghost)
#         with subtests.test(rank=rank):
#             quantity.np.testing.assert_array_equal(quantity.data, array)
