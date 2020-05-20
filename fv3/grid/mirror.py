from ..utils.global_constants import RADIUS, RIGHT_HAND_GRID, N_TILES
import math

__all__ = ["mirror_grid"]


def mirror_grid(grid_global, ng: int, npx: int, npy: int, np):
    # first fix base region
    nreg = 0
    for j in range(0, math.ceil(npy / 2)):
        for i in range(0, math.ceil(npx / 2)):
            x1 = 0.25 * (
                np.abs(grid_global[ng + i, ng + j, 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 0, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg])
            )
            grid_global[ng + i, ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + j, 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + j, 0, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + npy - (j + 1), 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg]
            )

            y1 = 0.25 * (
                np.abs(grid_global[ng + i, ng + j, 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 1, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg])
            )
            grid_global[ng + i, ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + j, 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + j, 1, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + npy - (j + 1), 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg]
            )

            # force dateline/greenwich-meridion consitency
            if npx % 2 != 0:
                if i == ng + (npx - 1) // 2:
                    grid_global[ng + i, ng + j, 0, nreg] = 0.0
                    grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = 0.0

    for nreg in range(1, N_TILES):
        for j in range(0, npy):
            x1 = grid_global[ng : ng + npx, ng + j, 0, 0]
            y1 = grid_global[ng : ng + npx, ng + j, 1, 0]
            z1 = RADIUS + 0.0 * x1

            if nreg == 1:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
            elif nreg == 2:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force North Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if i == (npx - 1) // 2 and i == j:
                        x2[:] = 0.0
                        y2[:] = np.pi / 2.0
                    if j == (npy - 1) // 2 and i < (npx - 1) // 2:
                        x2[:] = 0.0
                    if j == (npy - 1) // 2 and i > (npx - 1) // 2:
                        x2[:] = np.pi
            elif nreg == 3:
                ang = -180.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == (npy - 1) // 2:
                        x2[:] = np.pi
            elif nreg == 4:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    3, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
            elif nreg == 5:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2, [x1, y1, z1], ang, np, degrees=True, convert=True
                )
                ang = 0.0
                x2, y2, z2 = _rot_3d(
                    3, [x2, y2, z2], ang, np, degrees=True, convert=True
                )
                # force South Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if i == (npx - 1) // 2 and i == j:
                        x2[:] = 0.0
                        y2[:] = -np.pi / 2.0
                    if i == (npx - 1) // 2 and j > (npy - 1) // 2:
                        x2[:] = 0.0
                    if i == (npx - 1) // 2 and j < (npy - 1) // 2:
                        x2[:] = np.pi

            grid_global[ng : ng + npx, ng + j, 0, nreg] = x2
            grid_global[ng : ng + npx, ng + j, 1, nreg] = y2

    return grid_global


def _rot_3d(self, axis, p, angle, np, degrees=False, convert=False):

    if convert:
        p1 = _spherical_to_cartesian(p, np)
    else:
        p1 = p

    if degrees:
        angle = np.deg2rad(angle)

    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 1:
        x2 = p1[0]
        y2 = c * p1[1] + s * p1[2]
        z2 = -s * p1[1] + c * p1[2]
    elif axis == 2:
        x2 = c * p1[0] - s * p1[2]
        y2 = p1[1]
        z2 = s * p1[0] + c * p1[2]
    elif axis == 3:
        x2 = c * p1[0] + s * p1[1]
        y2 = -s * p1[0] + c * p1[1]
        z2 = p1[2]
    else:
        assert False, "axis must be in [1,2,3]"

    if convert:
        p2 = _cartesian_to_spherical([x2, y2, z2], np)
    else:
        p2 = [x2, y2, z2]

    return p2


def _spherical_to_cartesian(p, np):
    lon, lat, r = p
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    if RIGHT_HAND_GRID:
        z = r * np.sin(lat)
    else:
        z = -r * np.sin(lat)
    return [x, y, z]


def _cartesian_to_spherical(p, np):
    x, y, z = p
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.where(np.abs(x) + np.abs(y) < 1.0e-10, 0.0, np.arctan2(y, x))
    if RIGHT_HAND_GRID:
        lat = np.arcsin(z / r)
    else:
        lat = np.arccos(z / r) - np.pi / 2.0
    return [lon, lat, r]
