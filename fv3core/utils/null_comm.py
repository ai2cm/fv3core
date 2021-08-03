class NullAsyncResult:
    def wait(self):
        pass


class NullComm:
    """
    A class with a subset of the mpi4py Comm API, but which does nothing.
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
        pass

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        pass

    def Send(self, sendbuf, dest, **kwargs):
        pass

    def Isend(self, sendbuf, dest, **kwargs):
        return NullAsyncResult()

    def Recv(self, recvbuf, source, **kwargs):
        pass

    def Irecv(self, recvbuf, source, **kwargs):
        return NullAsyncResult()
