import fv3util


def halo_update(storage, comm, layout):
    quantity = fv3util.Quantity.from_gt4py_storage(storage)
    partitioner = fv3util.CubedSpherePartitioner(
        fv3util.TilePartitioner(layout)
    )
    communicator = fv3util.CubedSphereCommunicator(
        comm, partitioner
    )
    communicator.start_halo_update(quantity)
    communicator.finish_halo_update(quantity)
