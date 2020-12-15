import sys

import mpi4py
import yaml

import fv3core
import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3gfs.util as util


# sys.path.append("/usr/local/serialbox/python")
sys.path.append("/home/tobiasw/work/fv3core/tests")
import serialbox  # noqa: E402
import translate  # noqa: E402


if __name__ == "__main__":
    # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    # fv3core specific setup
    fv3core.set_backend("numpy")
    fv3core.set_rebuild(False)

    # namelist setup
    spec.set_namelist(
        "/home/tobiasw/work/fv3core/test_data/c12_6ranks_standard/input.nml"
    )
    nml2 = yaml.safe_load(
        open(
            "/home/tobiasw/work/fv3core/examples/wrapped/"
            "config/c12_6ranks_standard.yml",
            "r",
        )
    )["namelist"]

    # set up of helper structures
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        "/home/tobiasw/work/fv3core/test_data/c12_6ranks_standard",
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
    grid = translate.TranslateGrid(grid_data, rank).python_grid()
    spec.set_grid(grid)

    # set up grid-dependent helper structures
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    communicator = util.CubedSphereCommunicator(comm, partitioner)

    # create a state from serialized data
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    driver_object = translate.TranslateFVDynamics([grid])
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    input_data["comm"] = communicator
    state = driver_object.state_from_inputs(input_data)

    # Run the dynamics
    # for i in range(wrapper.get_step_count()):
    for i in range(1):
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
            raise Exception("this is not supported")
            # state["eastward_wind_tendency"] = u_tendency
            # state["northward_wind_tendency"] = v_tendency
            # fv3core.fv_subgridz(state, n_tracers, dt_atmos)

    # do we want to write out?
    # io.write_state(state, "outstate_{0}.nc".format(rank))
