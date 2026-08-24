import numpy as np

from preprocess.RealSenseGeometry import (
    backproject_pixels,
    project_points,
    transform_points,
)


def test_project_backproject_roundtrip():
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    uv = np.array([[320.0, 240.0], [420.0, 290.0]])
    z = np.array([1.0, 2.0])

    xyz = backproject_pixels(uv, z, K)
    uv2 = project_points(xyz, K)
    assert np.allclose(uv, uv2, atol=1e-8)


def test_transform():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    p = np.array([[0.1, 0.2, 0.3]])
    assert np.allclose(
        transform_points(T, p),
        [[1.1, 2.2, 3.3]],
    )


if __name__ == "__main__":
    test_project_backproject_roundtrip()
    test_transform()
    print("RealSense frontend geometry smoke test: PASS")
