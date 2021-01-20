import json
import pathlib
from argparse import ArgumentParser
from datetime import datetime
from statistics import mean, median

import git
import mpi4py
import serialbox
import yaml

import fv3core
import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.testing
import fv3gfs.util as util


if __name__ == "__main__":
    t0 = mpi4py.MPI.Wtime()

    usage = "usage: python %(prog)s <data_dir> <namelist_path> <timesteps>"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "namelist_path",
        type=str,
        action="store",
        help="path to the namelist",
    )
    parser.add_argument(
        "time_step",
        type=int,
        action="store",
        help="number of timesteps to execute",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    time_step = args.time_step
    namelist_path = args.namelist_path

    # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    # fv3core specific setup
    fv3core.set_backend("numpy")
    fv3core.set_rebuild(False)

    # namelist setup
    spec.set_namelist(data_dir + "/input.nml")

    nml2 = yaml.safe_load(
        open(
            namelist_path,
            "r",
        )
    )["namelist"]

    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_dir,
        "Generator_rank" + str(rank),
    )
    cube_comm = util.CubedSphereCommunicator(
        comm, util.CubedSpherePartitioner.from_namelist(nml2)
    )

    # get grid from serialized data
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = fv3core.testing.TranslateGrid(grid_data, rank).python_grid()
    spec.set_grid(grid)

    # set up grid-dependent helper structures
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    driver_object = fv3core.testing.TranslateFVDynamics([grid])
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    input_data["comm"] = communicator
    state = driver_object.state_from_inputs(input_data)

    # warm-up timestep
    fv_dynamics.fv_dynamics(
        state,
        communicator,
        input_data["consv_te"],
        input_data["do_adiabatic_init"],
        input_data["bdt"],
        input_data["ptop"],
        input_data["n_split"],
        input_data["ks"],
    )
    if spec.namelist.fv_sg_adj > 0:
        pass
        # raise Exception("this is not supported")
        # state["eastward_wind_tendency"] = u_tendency
        # state["northward_wind_tendency"] = v_tendency
        # fv3core.fv_subgridz(state, n_tracers, dt_atmos)

    t1 = mpi4py.MPI.Wtime()
    # Run the dynamics
    for i in range(time_step - 1):
        fv_dynamics.fv_dynamics(
            state,
            communicator,
            input_data["consv_te"],
            input_data["do_adiabatic_init"],
            input_data["bdt"],
            input_data["ptop"],
            input_data["n_split"],
            input_data["ks"],
        )
        if spec.namelist.fv_sg_adj > 0:
            pass
            # raise Exception("this is not supported")
            # state["eastward_wind_tendency"] = u_tendency
            # state["northward_wind_tendency"] = v_tendency
            # fv3core.fv_subgridz(state, n_tracers, dt_atmos)

    # collect times and output simple statistics
    t2 = mpi4py.MPI.Wtime()
    elapsed = t2 - t1
    init_times = t1 - t0
    alltimes = comm.gather(elapsed, root=0)
    init_times = comm.gather(init_times, root=0)
    if comm.Get_rank() == 0:
        now = datetime.now()
        sha = git.Repo(
            pathlib.Path(__file__).parent.absolute(), search_parent_directories=True
        ).head.object.hexsha
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        filename = now.strftime("%Y-%m-%d-%H-%M-%S")
        experiment = {}
        experiment["setup"] = {}
        experiment["setup"]["experiment time"] = dt_string
        experiment["setup"]["data set"] = "baroclinic"  # nml2
        experiment["setup"]["timesteps"] = time_step
        experiment["setup"]["timesteps"] = time_step
        experiment["setup"]["hash"] = sha
        experiment["setup"]["version"] = "python"

        experiment["times"] = {}
        experiment["times"]["total"] = {}
        experiment["times"]["init"] = {}
        experiment["times"]["init"]["minimum"] = min(init_times)
        experiment["times"]["init"]["maximum"] = max(init_times)
        experiment["times"]["init"]["median"] = median(init_times)
        experiment["times"]["init"]["mean"] = mean(init_times)
        experiment["times"]["main"] = {}
        experiment["times"]["cleanup"] = {}
        experiment["times"]["main"]["minimum"] = min(alltimes)
        experiment["times"]["main"]["maximum"] = max(alltimes)
        experiment["times"]["main"]["median"] = median(alltimes)
        experiment["times"]["main"]["mean"] = mean(alltimes)

        with open(filename + ".json", "w") as outfile:
            json.dump(experiment, outfile, sort_keys=True, indent=4)
