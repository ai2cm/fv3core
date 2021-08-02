#!/usr/bin/env python3
import copy
from types import SimpleNamespace

import click

import dace

import numpy as np
from fv3core.decorators import computepath_function
import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config

import fv3gfs
import serialbox
from fv3core.stencils.dyn_core import AcousticDynamics


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


def read_input_data(grid, serializer):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(grid, input_data):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return statevars


def run(data_directory, halo_update, backend, time_steps, reference_run):
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)



    state = get_state_from_input(grid, input_data)

    acoutstics_object = AcousticDynamics(
        None,
        spec.namelist,
        input_data["ak"],
        input_data["bk"],
        input_data["pfull"],
        input_data["phis"],
    )
    state.__dict__.update(acoutstics_object._temporaries)

    @computepath_function
    def iterate(state: dace.constant, time_steps):
        # @Linus: make this call a dace program
        for _ in range(time_steps):
            acoutstics_object(state, insert_temporaries=False)

    if reference_run:
        iterate(state, time_steps)
    else:
        from fv3core.utils.global_config import set_dacemode
        set_dacemode(True)

        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

        try:
            iterate(state, time_steps)
        finally:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print("\n".join(s.getvalue().split("\n")[:50]))
            set_dacemode(False)


    return state

@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default=1, type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--halo_update/--no-halo_update", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    halo_update: bool,
):
    state = run(data_directory, halo_update, time_steps=time_steps, backend=backend, reference_run=False)
    ref_state = run(data_directory, halo_update, time_steps=time_steps, backend='numpy', reference_run=True)

    for name, ref_value in ref_state.__dict__.items():

        if name in {'mfxd', 'mfyd'}:
            continue
        value = state.__dict__[name]
        if isinstance(ref_value, fv3gfs.util.quantity.Quantity):
            ref_value = ref_value.storage
        if isinstance(value, fv3gfs.util.quantity.Quantity):
            value = value.storage
        if hasattr(value, 'shape') and len(value.shape) == 3:
            value = value[1:-1, 1:-1, :]
            ref_value = ref_value[1:-1, 1:-1, :]
        np.testing.assert_allclose(np.asarray(ref_value), np.asarray(value), err_msg=name)


if __name__ == "__main__":
    driver()
