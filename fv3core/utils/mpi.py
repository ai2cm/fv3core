try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from typing import Any


def is_parallel_context() -> bool:
    """Checks if we are in a parallel context, meaing the size of the
    world communicatior is greater than 1.

    Returns:
        bool: if the context is parallel
    """
    return bool(MPI is not None and MPI.COMM_WORLD.Get_size() > 1)


def get_size() -> int:
    """Get the size of the MPI COMM_WORLD communicator if it is defined.

    Returns:
        int: size
    """
    return MPI.COMM_WORLD.Get_size() if MPI else 0


def get_rank() -> int:
    """Get the current rank in the MPI COMM_WORLD communicator if it is defined.

    Returns:
        int: rank
    """
    return MPI.COMM_WORLD.Get_rank() if is_parallel_context() else -1


def bcast(data) -> None:
    """A wrapper for a broadcast in the MPI COMM_WORLD communicator from rank 0.

    Args:
        data: The data to broadcast
    """
    return MPI.COMM_WORLD.bcast(data, root=0)


def send(dest: int, tag: int = 0) -> None:
    """A wrapper for a dummy send in the MPI COMM_WORLD communicator
    used for blocking other threads from advancing.

    Args:
        dest: The destination rank
        tag: The tag that the MPI send should have
    """
    MPI.COMM_WORLD.send(1, dest=dest, tag=tag)


def recv(source: int, tag: int = 0) -> Any:
    """A wrapper for a dummy receive in the MPI COMM_WORLD communicator used for
    blocking other threads from advancing.

    Args:
        source: The source rank
        tag: The tag that the MPI send will have
    Returns:
        Forwadrd the recv return value
    """
    buff = bytearray(1 << 10)
    return MPI.COMM_WORLD.recv(buff, source=source, tag=tag)
