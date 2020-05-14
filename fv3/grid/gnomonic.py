from ..constants import pi
import numpy as np


def gnomonic_grid(grid_type: int, lon, lat):
    """Apply gnomonic grid to lon and lat arrays.

    Args:
        grid_type: ???
        lon: array with dimensions [x, y]
        lat: array with dimensions [x, y]
    """
    _check_shapes(lon, lat)
    if grid_type == 0:
        gnomonic_ed(lon, lat)
    elif grid_type == 1:
        gnomonic_dist(lon, lat)
    elif grid_type == 2:
        gnomonic_angl(lon, lat)
    if grid_type < 3:
        symm_ed(lon, lat)
        lon = lon - pi


def _check_shapes(lon, lat):
    if len(lon.shape) != 2:
        raise ValueError(f"longitude must be 2D, has shape {lon.shape}")
    elif len(lat.shape) != 2:
        raise ValueError(f"latitude must be 2D, has shape {lat.shape}")
    elif lon.shape[0] != lon.shape[1]:
        raise ValueError(f"longitude must be square, has shape {lon.shape}")
    elif lat.shape[0] != lat.shape[1]:
        raise ValueError(f"latitude must be square, has shape {lat.shape}")
    elif lon.shape[0] != lat.shape[0]:
        raise ValueError(
            "longitude and latitude must have same shape, but they are "
            f"{lon.shape} and {lat.shape}"
        )


def gnomonic_ed(lon, lat):
    im = lon.shape[0]
    alpha = np.arcsin(3 ** -.5)

    dely = 2.0 * alpha / float(im)

    pp = np.empty((3, im + 1, im + 1))

    for j in range(0, im + 1):
        lon[0, j] = 0.75 * np.pi  # West edge
        lon[im, j] = 1.25 * np.pi  # East edge
        lat[0, j] = -alpha + dely * float(j)  # West edge
        lat[im, j] = lat[0, j]  # East edge

    # Get North-South edges by symmetry
    for i in range(1, im):
        lon[i, 0], lat[i, 0] = _mirror_latlon(
            lon[0, 0], lat[0, 0], lon[im, im], lat[im, im], lon[0, i], lat[0, i]
        )
        lon[i, im] = lon[i, 0]
        lat[i, im] = -lat[i, 0]

    # set 4 corners
    pp[:, 0, 0] = _latlon2xyz(lon[0, 0], lat[0, 0])
    pp[:, im, 0] = _latlon2xyz(lon[im, 0], lat[im, 0])
    pp[:, 0, im] = _latlon2xyz(lon[0, im], lat[0, im])
    pp[:, im, im] = _latlon2xyz(lon[im, im], lat[im, im])

    # map edges on the sphere back to cube: intersection at x = -1/sqrt(3)
    i = 0
    for j in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j])
        pp[1, i, j] = -pp[1, i, j] * (3 ** -.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -.5) / pp[0, i, j]

    j = 0
    for i in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j])
        pp[1, i, j] = -pp[1, i, j] * (3 ** -.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -.5) / pp[0, i, j]

    pp[0, :, :] = -(3 ** -.5)

    for j in range(1, im + 1):
        # copy y-z face of the cube along j=0
        pp[1, 1:, j] = pp[1, 1:, 0]
        # copy along i=0
        pp[2, 1:, j] = pp[2, 0, j]

    pp, lon, lat = _cart_to_latlon(im + 1, pp, lon, lat)


def _latlon2xyz(lon, lat):
    """map (lon, lat) to (x, y, z)"""

    e1 = np.cos(lat) * np.cos(lon)
    e2 = np.cos(lat) * np.sin(lon)
    e3 = np.sin(lat)

    return [e1, e2, e3]


def _cart_to_latlon(im, q, xs, ys):
    """map (x, y, z) to (lon, lat)"""

    esl = 1.0e-10

    for j in range(im):
        for i in range(im):
            p = q[:, i, j]
            dist = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
            p = p / dist

            if np.abs(p[0]) + np.abs(p[1]) < esl:
                lon = 0.0
            else:
                lon = np.arctan2(p[1], p[0])  # range [-pi, pi]

            if lon < 0.0:
                lon = 2.0 * np.pi + lon

            lat = np.arcsin(p[2])

            xs[i, j] = lon
            ys[i, j] = lat

            q[:, i, j] = p

    return q, xs, ys


def _mirror_latlon(lon1, lat1, lon2, lat2, lon0, lat0):

    p0 = _latlon2xyz(lon0, lat0)
    p1 = _latlon2xyz(lon1, lat1)
    p2 = _latlon2xyz(lon2, lat2)
    nb = _vect_cross(p1, p2)

    pdot = np.sqrt(nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2)
    nb = nb / pdot

    pdot = p0[0] * nb[0] + p0[1] * nb[1] + p0[2] * nb[2]
    pp = p0 - 2.0 * pdot * nb

    lon3 = np.empty((1, 1))
    lat3 = np.empty((1, 1))
    pp3 = np.empty((3, 1, 1))
    pp3[:, 0, 0] = pp
    _cart_to_latlon(1, pp3, lon3, lat3)

    return lon3[0, 0], lat3[0, 0]


def _vect_cross(p1, p2):
    return [
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p1[1] * p2[0],
    ]


def gnomonic_dist(lon, lat):
    raise NotImplementedError()


def gnomonic_angl(lon, lat):
    raise NotImplementedError()


def symm_ed(lon, lat):
    pass
