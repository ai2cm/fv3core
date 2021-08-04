class NullAsyncResult:
    def __init__(self, recvbuf=None):
        self._recvbuf = recvbuf

    def wait(self):
        if self._recvbuf is not None:
            self._recvbuf[:] = 0.0


class NullComm:
    """
    A class with a subset of the mpi4py Comm API, but which
    'receives' zeros instead of using MPI.
    """

    def __init__(self, rank, total_ranks):
        self.rank = rank
        self.total_ranks = total_ranks

    def __repr__(self):
        return f"NullComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.total_ranks

    def bcast(self, value, root=0):
        return value

    def barrier(self):
        return

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        if recvbuf is not None:
            recvbuf[:] = 0.0

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        if recvbuf is not None:
            recvbuf[:] = 0.0

    def Send(self, sendbuf, dest, **kwargs):
        pass

    def Isend(self, sendbuf, dest, **kwargs):
        return NullAsyncResult()

    def Recv(self, recvbuf, source, **kwargs):
        recvbuf[:] = 0.0

    def Irecv(self, recvbuf, source, **kwargs):
        return NullAsyncResult(recvbuf)
