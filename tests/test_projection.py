import numpy as np
import pytest

from dscamera import DSCamera


@pytest.fixture
def cam():
    intrinsic = {
        "fx": 120.0,
        "fy": 120.0,
        "cx": 320.0,
        "cy": 240.0,
        "xi": -0.1,
        "alpha": 0.55,
    }
    img_size = (480, 640)
    return DSCamera(intrinsic=intrinsic, img_size=img_size)


def test_unproj_proj(cam):
    # Generate random data
    u = np.random.rand(100) * cam.w
    v = np.random.rand(100) * cam.h

    # Unprojection and projection
    unproj_pts, valid_unproj = cam.cam2world([u, v])
    proj_pts, valid_proj = cam.world2cam(unproj_pts)
    valid_mask = valid_unproj * valid_proj
    valid_pts = proj_pts[valid_mask]
    assert np.allclose(u[valid_mask], valid_pts[..., 0])
    assert np.allclose(v[valid_mask], valid_pts[..., 1])


def test_proj_unproj(cam):
    x = 2 * np.random.rand(100) - 1
    y = 2 * np.random.rand(100) - 1
    z = 2 * np.random.rand(100) - 1
    point3D = np.stack([x, y, z], axis=-1)
    # Remove invalid points:(0,0,0)
    invalid = np.all(point3D == 0.0, axis=-1)
    point3D = point3D[~invalid]
    # Unit vector
    point3D /= np.linalg.norm(point3D, axis=-1, keepdims=True)

    # Projection and unprojection
    proj_pts, valid_proj = cam.world2cam(point3D)
    u, v = proj_pts[:, 0], proj_pts[:, 1]
    unproj_pts, valid_unproj = cam.cam2world([u, v])
    valid_mask = valid_unproj * valid_proj
    valid_pts = unproj_pts[valid_mask]
    assert np.allclose(point3D[valid_mask], valid_pts)


def test_optical_center(cam):
    # Edge case: optical center
    pts, valid = cam.world2cam(np.array([0, 0, 1]))
    assert valid
    assert np.allclose(pts, np.array([cam.cx, cam.cy]))

    pts, valid = cam.cam2world([cam.cx, cam.cy])
    assert valid
    assert np.allclose(pts, np.array([0, 0, 1]))
