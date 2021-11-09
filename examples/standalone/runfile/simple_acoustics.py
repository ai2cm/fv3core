#!/usr/bin/env python3
from types import SimpleNamespace

import click
import dace
import serialbox
from mpi4py import MPI

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
from fv3core.decorators import computepath_function
from fv3core.stencils.dyn_core import AcousticDynamics
import fv3gfs.util as util


def set_up_namelist(data_directory: str) -> None:
    spec.set_namelist(data_directory + "/input.nml")


def initialize_serializer(data_directory: str, rank: int = 0) -> serialbox.Serializer:
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )


def read_grid(
    serializer: serialbox.Serializer, rank: int = 0
) -> fv3core.testing.TranslateGrid:
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    return fv3core.testing.TranslateGrid(grid_data, rank).python_grid()


def initialize_fv3core(backend: str, do_halo_updates: bool) -> None:
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)
    global_config.set_do_halo_exchange(do_halo_updates)


def read_input_data(grid, serializer) -> dict:
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(grid, input_data) -> SimpleNamespace:
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return statevars


def run(
    data_directory: str,
    halo_update: bool,
    backend: str,
    time_steps: int,
    sdfg_path: str,
) -> SimpleNamespace:
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)
    state = get_state_from_input(grid, input_data)

    # set the communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    acoutstics_object = AcousticDynamics(
        None,
        spec.namelist,
        input_data["ak"],
        input_data["bk"],
        input_data["pfull"],
        input_data["phis"],
    )
    state.__dict__.update(acoutstics_object._temporaries)

    sdfg_paths = {
        "sdfg1_data_init": None,
        "sdfg2_c_sw": None,
        "sdfg3_c_grid": None,
        "sdfg4_d_sw": None,
        "sdfg5_updatedzd": None,
        "sdfg6_pk3_halo": None,
        "sdfg8_nh_p_grad": None,
        "sdfg9_hyperdiffusion": None,
    }
    if sdfg_path != "":
        if MPI.COMM_WORLD.Get_size() > 1:
            sdfg_path_base = sdfg_path + str(rank) + "/dacecache/"
        else:
            sdfg_path_base = sdfg_path + "/dacecache/"
        for key, value in sdfg_paths.items():
            sdfg_paths[key] = f"{sdfg_path_base}/{key}"
    else:
        for key, value in sdfg_paths.items():
            sdfg_paths[key] = None

    @computepath_function(skip_dacemode=True)
    def iterate(state: dace.constant, time_steps):
        for _ in range(time_steps):
            acoutstics_object(state, insert_temporaries=False, sdfg_paths=sdfg_paths)

    import time

    iterate(state, 1)
    start = time.time()
    try:
        iterate(state, time_steps)
    finally:
        print(
            f"Total {backend} time on rank {rank} for {time_steps} steps:",
            time.time() - start,
        )

    comm.Barrier()
    return state


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default=1, type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--halo_update/--no-halo_update", default=False)
@click.argument("sdfg_path", required=False, default="")
def driver(
    data_directory: str,
    time_steps: int,
    backend: str,
    halo_update: bool,
    sdfg_path: str,
):

    state = run(
        data_directory,
        halo_update,
        time_steps=time_steps,
        backend=backend,
        sdfg_path=sdfg_path,
    )


if __name__ == "__main__":
    driver()
