import os

if "MPI_SPOOF" in os.environ:

    class MPI:
        COMM_WORLD = None

        class Comm:  # For type hints
            pass

    if MPI.COMM_WORLD is None:  # Run once
        from fv3core.utils.null_comm import NullComm

        rank, total_ranks = os.environ["MPI_SPOOF"].split(",")
        MPI.COMM_WORLD = NullComm(int(rank), int(total_ranks), 1.0)
        print(f"[NullComm] Spoofing MPI rank {rank} out of {total_ranks}")
else:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
