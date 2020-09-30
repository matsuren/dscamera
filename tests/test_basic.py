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


def test_eq():
    intrinsic = {
        "fx": 120.0,
        "fy": 120.0,
        "cx": 320.0,
        "cy": 240.0,
        "xi": -0.1,
        "alpha": 0.55,
    }
    img_size = (480, 640)
    # Equal
    cam1 = DSCamera(intrinsic=intrinsic, img_size=img_size)
    cam2 = DSCamera(intrinsic=intrinsic, img_size=img_size)
    assert cam1 == cam2

    # Not equal
    cam1 = DSCamera(intrinsic=intrinsic, img_size=img_size)
    intrinsic["xi"] += 0.1
    cam2 = DSCamera(intrinsic=intrinsic, img_size=img_size)
    assert cam1 != cam2


def test_img_size(cam):
    # Get image size
    img_size = cam.img_size
    assert img_size[0] == cam.h
    assert img_size[1] == cam.w

    # Set image size
    img_size = (100, 200)
    cam.img_size = img_size
    assert cam.h == img_size[0]
    assert cam.w == img_size[1]


def test_intrinsic(cam):
    # Get intrinsic
    intrinsic = cam.intrinsic
    assert intrinsic["fx"] == cam.fx
    assert intrinsic["fy"] == cam.fy
    assert intrinsic["cx"] == cam.cx
    assert intrinsic["cy"] == cam.cy
    assert intrinsic["xi"] == cam.xi
    assert intrinsic["alpha"] == cam.alpha

    # Set intrinsic
    intrinsic["xi"] = 0.2
    intrinsic["alpha"] = 0.1
    cam.intrinsic = intrinsic
    assert cam.xi == 0.2
    assert cam.alpha == 0.1
