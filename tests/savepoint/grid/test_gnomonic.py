import numpy as np
import pytest

from fv3.grid.gnomonic import (
    _latlon2xyz,
    _xyz2latlon,
    great_circle_dist,
    spherical_angle,
)
from fv3.utils.global_constants import PI


@pytest.mark.parametrize(
    "lon, lat, radius, axis, reference",
    [
        pytest.param(
            np.array([0, 0]), np.array([0, PI]), 1, 0, np.array([PI]), id="pole_to_pole"
        ),
        pytest.param(
            np.array([0, 0]),
            np.array([0, PI]),
            2,
            0,
            np.array([2 * PI]),
            id="pole_to_pole_greater_radius",
        ),
        pytest.param(
            np.array([0.3, 0.3]),  # arbitrary longitude
            np.array([PI / 2, PI]),
            1,
            0,
            np.array([PI / 2]),
            id="equator_to_pole",
        ),
        pytest.param(
            np.array([0.3, 0.5]),  # arbitrary longitude
            np.array([PI / 2, PI]),
            1,
            0,
            np.array([PI / 2]),
            id="equator_to_pole_different_lons",
        ),
        pytest.param(
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [PI, PI]]),
            1,
            0,
            np.array([[PI, PI]]),
            id="pole_to_pole_2d_first_dim",
        ),
        pytest.param(
            np.array([[0, 0], [0, 0]]),
            np.array([[0, PI], [0, PI]]),
            1,
            1,
            np.array([[PI], [PI]]),
            id="pole_to_pole_2d_second_dim",
        ),
    ],
)
def test_great_circle_dist(lon, lat, radius, axis, reference):
    result = great_circle_dist(lon, lat, radius, np, axis)
    np.testing.assert_array_almost_equal(result, reference)


@pytest.mark.parametrize(
    "lon, lat",
    [
        np.broadcast_arrays(
            np.random.uniform(0, 2 * PI, 3)[:, None],
            np.random.uniform(-PI / 4, PI / 4, 3)[None, :],
        ),
        np.broadcast_arrays(
            np.random.uniform(0, 2 * PI, 5)[:, None],
            np.random.uniform(-PI / 4, PI / 4, 5)[None, :],
        ),
    ],
)
def test_latlon2xyz_xyz2latlon_is_identity(lon, lat):
    x, y, z = _latlon2xyz(lon, lat, np)
    lon_out, lat_out = _xyz2latlon(x, y, z, np)
    np.testing.assert_array_almost_equal(lat_out, lat)
    np.testing.assert_array_almost_equal(lon_out, lon)


@pytest.mark.parametrize(
    "p_center, p2, p3, angle",
    [
        pytest.param(
            np.array([[1, 0, 0]]),
            np.array([[0, 1, 0]]),
            np.array([[0, 0, 1]]),
            np.array([PI / 2]),
            id="cube_face_centers",
        ),
        pytest.param(
            np.array([[1, 0, 0]]),
            np.array([[0.01, 1, 0]]),
            np.array([[0.01, 0, 1]]),
            np.array([PI / 2]),
            id="cube_face_almost_centers",
        ),
        pytest.param(
            np.array([[1, 0, 0]]),
            np.array([[1, 0.1, 0]]),
            np.array([[1, 0, 0.1]]),
            np.array([PI / 2]),
            id="small_right_angle",
        ),
        pytest.param(
            np.array([[0, 1, 0]]),
            np.array([[1, 0, 0]]),
            np.array([[-1, 0, 0]]),
            np.array([PI]),
            id="straight_line",
        ),
    ],
)
def test_spherical_angle_easy_cases(p_center, p2, p3, angle):
    p_center = p_center / np.sqrt(np.sum(p_center ** 2, axis=-1))
    p2 = p2 / np.sqrt(np.sum(p2 ** 2, axis=-1))
    p3 = p3 / np.sqrt(np.sum(p3 ** 2, axis=-1))
    result = spherical_angle(p_center, p2, p3, np)
    np.testing.assert_array_equal(result, angle)


@pytest.mark.parametrize("angle", np.linspace(0, PI, 13))
def test_spherical_angle(angle):
    epsilon = 0.1
    p_center = np.array([[1, 0, 0]])
    p2 = np.array([[1, epsilon, 0]])
    p3 = np.array([[1, epsilon * np.cos(angle), epsilon * np.sin(angle)]])
    # normalize back onto sphere
    p_center = p_center / np.sqrt(np.sum(p_center ** 2, axis=-1))
    p2 = p2 / np.sqrt(np.sum(p2 ** 2, axis=-1))
    p3 = p3 / np.sqrt(np.sum(p3 ** 2, axis=-1))
    result = spherical_angle(p_center, p2, p3, np)
    np.testing.assert_array_almost_equal(result, angle)
