import tempfile

import pytest

from dscamera import DSCamera


@pytest.fixture
def cam():
    intrinsic = {
        "fx": 122.5,
        "fy": 121.7,
        "cx": 318.8,
        "cy": 235.7,
        "xi": -0.02,
        "alpha": 0.56,
    }
    img_size = (480, 640)
    return DSCamera(intrinsic=intrinsic, img_size=img_size)


def test_load_from_json(cam):
    msg = (
        '{"value0": {"intrinsics": [{"camera_type": "ds", '
        '"intrinsics": {"fx": 122.5, "fy": 121.7, "cx": 318.8, "cy": 235.7, '
        '"xi": -0.02, "alpha": 0.56}}], "resolution": [[640, 480]]}}'
    )
    # Tmp json file
    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, "w") as f:
            f.write(msg)

        # Load from json file
        cam_json = DSCamera(json_filename=tf.name)
        assert cam == cam_json


def test_eq():
    intrinsic = {
        "fx": 122.5,
        "fy": 120.0,
        "cx": 320.5,
        "cy": 240.1,
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


def test_hash(cam):
    # Create the same camera
    intrinsic = cam.intrinsic
    img_size = cam.img_size
    cam1 = DSCamera(intrinsic=intrinsic, img_size=img_size)

    # Dict
    table = {}
    table[cam] = "fixture_cam"
    assert table[cam1] == "fixture_cam"


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
