from ..utils.global_constants import PI
import contextlib
import copy


def gnomonic_grid(grid_type: int, lon, lat, np):
    """Apply gnomonic grid to lon and lat arrays.

    Args:
        grid_type: ???
        lon: longitude array with dimensions [x, y]
        lat: latitude array with dimensions [x, y]
    """
    _check_shapes(lon, lat)
    if grid_type == 0:
        gnomonic_ed(lon, lat, np)
    elif grid_type == 1:
        gnomonic_dist(lon, lat)
    elif grid_type == 2:
        gnomonic_angl(lon, lat)
    if grid_type < 3:
        symm_ed(lon, lat)
        lon[:] -= PI


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


def gnomonic_ed(lon, lat, np):
    im = lon.shape[0] - 1
    alpha = np.arcsin(3 ** -0.5)

    dely = 2.0 * alpha / float(im)

    pp = np.empty((3, im + 1, im + 1))

    for j in range(0, im + 1):
        lon[0, j] = 0.75 * PI  # West edge
        lon[im, j] = 1.25 * PI  # East edge
        lat[0, j] = -alpha + dely * float(j)  # West edge
        lat[im, j] = lat[0, j]  # East edge

    # Get North-South edges by symmetry
    for i in range(1, im):
        lon[i, 0], lat[i, 0] = _mirror_latlon(
            lon[0, 0], lat[0, 0], lon[im, im], lat[im, im], lon[0, i], lat[0, i], np
        )
        lon[i, im] = lon[i, 0]
        lat[i, im] = -lat[i, 0]

    # set 4 corners
    pp[:, 0, 0] = _latlon2xyz(lon[0, 0], lat[0, 0], np)
    pp[:, im, 0] = _latlon2xyz(lon[im, 0], lat[im, 0], np)
    pp[:, 0, im] = _latlon2xyz(lon[0, im], lat[0, im], np)
    pp[:, im, im] = _latlon2xyz(lon[im, im], lat[im, im], np)

    # map edges on the sphere back to cube: intersection at x = -1/sqrt(3)
    i = 0
    for j in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j], np)
        pp[1, i, j] = -pp[1, i, j] * (3 ** -0.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -0.5) / pp[0, i, j]

    j = 0
    for i in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j], np)
        pp[1, i, j] = -pp[1, i, j] * (3 ** -0.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -0.5) / pp[0, i, j]

    pp[0, :, :] = -(3 ** -0.5)

    for j in range(1, im + 1):
        # copy y-z face of the cube along j=0
        pp[1, 1:, j] = pp[1, 1:, 0]
        # copy along i=0
        pp[2, 1:, j] = pp[2, 0, j]

    _cart_to_latlon(im + 1, pp, lon, lat, np)


def _corner_to_center_mean(corner_array):
    """Given a 2D array on cell corners, return a 2D array on cell centers with the
    mean value of each of the corners."""
    return 0.25 * (
        corner_array[1:, 1:] +
        corner_array[:-1, :-1] +
        corner_array[1:, :-1] +
        corner_array[:-1, 1:]
    )


def normalize_vector(np, *vector_components):
    scale = 1 / sum(item ** 2 for item in vector_components) ** 0.5
    return (item * scale for item in vector_components)


def lat_lon_midpoint(lon1, lon2, lat1, lat2, np):
    x1, y1, z1 = _latlon2xyz(lon1, lat1, np)
    x2, y2, z2 = _latlon2xyz(lon2, lat2, np)
    x = 0.5 * (x1 + x2)
    y = 0.5 * (y1 + y2)
    z = 0.5 * (z1 + z2)
    return _xyz2latlon(x, y, z, np)


def lat_lon_corner_to_cell_center(lon, lat, np):
    # just perform the mean in x-y-z space and convert back
    x, y, z = _latlon2xyz(lon, lat, np)
    x_center = _corner_to_center_mean(x)
    y_center = _corner_to_center_mean(y)
    z_center = _corner_to_center_mean(z)
    return _xyz2latlon(x_center, y_center, z_center, np)


def _latlon2xyz(lon, lat, np):
    """map (lon, lat) to (x, y, z)"""
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return normalize_vector(np, x, y, z)


def _xyz2latlon(x, y, z, np):
    """map (x, y, z) to (lon, lat)"""
    x, y, z = normalize_vector(np, x, y, z)
    lon = 0. * x
    nonzero_lon = np.abs(x) + np.abs(y) >= 1.0e-10
    lon[nonzero_lon] = np.arctan2(y[nonzero_lon], x[nonzero_lon])
    negative_lon = lon < 0.
    while np.any(negative_lon):
        lon[negative_lon] += 2 * PI
        negative_lon = lon < 0.
    lat = np.arcsin(z)

    return lon, lat


def _cart_to_latlon(im, q, xs, ys, np):
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
                lon = np.arctan2(p[1], p[0])  # range [-PI, PI]

            if lon < 0.0:
                lon = 2.0 * PI + lon

            lat = np.arcsin(p[2])

            xs[i, j] = lon
            ys[i, j] = lat

            q[:, i, j] = p


def _mirror_latlon(lon1, lat1, lon2, lat2, lon0, lat0, np):

    p0 = _latlon2xyz(lon0, lat0, np)
    p1 = _latlon2xyz(lon1, lat1, np)
    p2 = _latlon2xyz(lon2, lat2, np)
    nb = _vect_cross(p1, p2)

    pdot = np.sqrt(nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2)
    nb = nb / pdot

    pdot = p0[0] * nb[0] + p0[1] * nb[1] + p0[2] * nb[2]
    pp = p0 - 2.0 * pdot * nb

    lon3 = np.empty((1, 1))
    lat3 = np.empty((1, 1))
    pp3 = np.empty((3, 1, 1))
    pp3[:, 0, 0] = pp
    _cart_to_latlon(1, pp3, lon3, lat3, np)

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


def great_circle_beta(lon, lat, np, axis=0):
    """Returns the great-circle distance between points along the desired axis,
    as a fraction of the radius of the sphere."""
    lon, lat = np.broadcast_arrays(lon, lat)
    if len(lon.shape) == 1:
        case_1d = True
        # add singleton dimension so we can use the same indexing notation as n-d cases
        lon, lat = lon[:, None], lat[:, None]
    else:
        case_1d = False
    swap_dims = list(range(len(lon.shape)))
    swap_dims[axis], swap_dims[0] = swap_dims[0], swap_dims[axis]
    # below code computes distance along first axis, so we put the desired axis there
    lon, lat = lon.transpose(swap_dims), lat.transpose(swap_dims)
    result = (
        np.arcsin(
            np.sqrt(
                np.sin((lat[:-1, :] - lat[1:, :]) / 2.0) ** 2
                + np.cos(lat[:-1, :]) * np.cos(lat[1:, :]) * np.sin((lon[:-1, :] - lon[1:, :]) / 2.0) ** 2
            )
        )
        * 2.0
    )
    result = result.transpose(swap_dims)  # remember to swap back
    if case_1d:
        result = result[:, 0]  # remove the singleton dimension we added
    return result


def great_circle_dist(lon, lat, radius, np, axis=0):
    """Returns the great-circle distance between points along the desired axis."""
    return great_circle_beta(lon, lat, np, axis=axis) * radius


def _great_circle_beta(p1, p2, np):
    return (
        np.arcsin(
            np.sqrt(
                np.sin((p1[1] - p2[1]) / 2.0) ** 2
                + np.cos(p1[1]) * np.cos(p2[1]) * np.sin((p1[0] - p2[0]) / 2.0) ** 2
            )
        )
        * 2.0
    )


def _great_circle_dist(p1, p2, radius, np):
    return great_circle_beta(p1, p2, np) * radius


def get_area(lon, lat, radius, np):
    """
    Given latitude and longitude on cell corners, return the area of each cell.
    """
    xyz = np.concatenate(
        [arr[:, :, None] for arr in _latlon2xyz(lon, lat, np)], axis=-1
    )
    xyz /= np.sqrt(np.sum(xyz**2, axis=-1))[:, :, None]
    lower_left = (slice(None, -1), slice(None, -1), slice(None, None))
    lower_right = (slice(1, None), slice(None, -1), slice(None, None))
    upper_left = (slice(None, -1), slice(1, None), slice(None, None))
    upper_right = (slice(1, None), slice(1, None), slice(None, None))
    total_angle = spherical_angle(xyz[lower_left], xyz[lower_right], xyz[upper_left], np)
    for corners in (
            (lower_right, upper_right, lower_left),
            (upper_right, upper_left, lower_right),
            (upper_left, upper_right, lower_left)):
        new_angle = spherical_angle(
            xyz[corners[0]], xyz[corners[1]], xyz[corners[2]], np
        )
        total_angle += new_angle
    # total_angle = spherical_angle(xyz[upper_left], xyz[lower_left], xyz[upper_right], np)
    # for corners in (
    #         (upper_right, upper_left, lower_right),
    #         (lower_right, upper_right, lower_left),
    #         (lower_left, lower_right, upper_left)):
    #     total_angle += spherical_angle(
    #         xyz[corners[0]], xyz[corners[1]], xyz[corners[2]], np
    #     )
    return (total_angle - 2 * PI) * radius ** 2


def set_corner_area_to_triangle_area(lon, lat, area, radius, np):
    """
    Given latitude and longitude on cell corners, and an array of cell areas, set the
    four corner areas to the area of the inner triangle at those corners.
    """
    xyz = np.concatenate(
        [arr[:, :, None] for arr in _latlon2xyz(lon, lat, np)], axis=-1
    )
    xyz /= np.sqrt(np.sum(xyz**2, axis=-1))[:, :, None]
    lower_left = xyz[(slice(None, -1), slice(None, -1), slice(None, None))]
    lower_right = xyz[(slice(1, None), slice(None, -1), slice(None, None))]
    upper_left = xyz[(slice(None, -1), slice(1, None), slice(None, None))]
    upper_right = xyz[(slice(1, None), slice(1, None), slice(None, None))]
    area[0, 0] = get_triangle_area(upper_left[0, 0], upper_right[0, 0], lower_right[0, 0], radius, np)
    area[-1, 0] = get_triangle_area(upper_right[-1, 0], upper_left[-1, 0], lower_left[-1, 0], radius, np)
    area[-1, -1] = get_triangle_area(lower_right[-1, -1], lower_left[-1, -1], upper_left[-1, -1], radius, np)
    area[0, -1] = get_triangle_area(lower_left[0, -1], lower_right[0, -1], upper_right[0, -1], radius, np)


def get_triangle_area(p1, p2, p3, radius, np):
    """
    Given three point arrays whose last dimensions are x/y/z, return an array of
    spherical triangle areas.
    """

    total_angle = spherical_angle(p1, p2, p3, np)
    for q1, q2, q3 in (
            (p2, p3, p1),
            (p3, p1, p2)):
        total_angle += spherical_angle(q1, q2, q3, np)
    # total_angle = spherical_angle(xyz[upper_left], xyz[lower_left], xyz[upper_right], np)
    # for corners in (
    #         (upper_right, upper_left, lower_right),
    #         (lower_right, upper_right, lower_left),
    #         (lower_left, lower_right, upper_left)):
    #     total_angle += spherical_angle(
    #         xyz[corners[0]], xyz[corners[1]], xyz[corners[2]], np
    #     )
    return (total_angle - PI) * radius ** 2


def spherical_angle(p_center, p2, p3, np):
    """
    Given ndarrays whose last dimension is x/y/z, compute the spherical angle between
    them according to:
!           p3
!         /
!        /
!       p_center ---> angle
!         \
!          \
!           p2
    """

# ! Vector P:
#    px = e1(2)*e2(3) - e1(3)*e2(2)
#    py = e1(3)*e2(1) - e1(1)*e2(3)
#    pz = e1(1)*e2(2) - e1(2)*e2(1)
# ! Vector Q:
#    qx = e1(2)*e3(3) - e1(3)*e3(2)
#    qy = e1(3)*e3(1) - e1(1)*e3(3)
#    qz = e1(1)*e3(2) - e1(2)*e3(1)
    p = np.cross(p_center, p2)
    q = np.cross(p_center, p3)
    # ddd = np.sum(p**2, axis=-1) * np.sum(q**2, axis=-1)
    # ddd_negative = ddd <= 0.
    # ddd = np.sum(p * q, axis=-1) / np.sqrt(ddd)
    # angle = np.arccos(ddd)
    # angle[ddd_negative] = 0.
    # angle[np.abs(ddd) > 1] = 0.5 * PI
    # angle[ddd < 0] = PI
    # return angle
    return np.arccos(np.sum(p * q, axis=-1) / np.sqrt(np.sum(p**2, axis=-1) * np.sum(q**2, axis=-1)))


#    ddd = (px*px+py*py+pz*pz)*(qx*qx+qy*qy+qz*qz)

#    if ( ddd <= 0.0d0 ) then
#         angle = 0.d0
#    else
#         ddd = (px*qx+py*qy+pz*qz) / sqrt(ddd)
#         if ( abs(ddd)>1.d0) then
#              angle = 2.d0*atan(1.0)    ! 0.5*pi
#            !FIX (lmh) to correctly handle co-linear points (angle near pi or 0)
#            if (ddd < 0.d0) then
#               angle = 4.d0*atan(1.0d0) !should be pi
#            else
#               angle = 0.d0 
#            end if
#         else
#              angle = acos( ddd )
#         endif
#    endif

#    spherical_angle = angle