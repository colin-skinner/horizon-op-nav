"""I hate having a generic `utils.py` but it doesn't need to be that complicated"""
import numpy as np
from numpy import sin, cos, pi

def circle_points(radius, N = 100, position = None):
    p = np.array([0,0]) if position is None else position
    t = np.linspace(0, pi * 2, N, endpoint=False)
    # print(p)
    # print(radius)
    # print(cos(t).shape)
    # print(radius * cos(t) + p[0])
    return np.array([radius * cos(t) + p[0], radius * sin(t) + p[1]])
    # x = radius * np.cos(t) + p[0]
    # y = radius * np.sin(t) + p[1]
    # return np.vstack((x, y)).T   # shape (N,2)

def camera_view(rho: np.ndarray, R: np.ndarray):
    """Returns what camera sees if perfectly aligned with planet along the camera's Z axis

    Args:
        rho (np.ndarray(3)): position of spacecraft in PLANET FRAME
        R (np.ndarray(3,3)): passive rotation from planet frame to camera frame

    Returns:
        r_c: View from camera TO planet in CAMERA FRAME
    """
    return R @ (-rho)

def noise(cov: np.ndarray):
    return np.random.multivariate_normal(
        mean = np.zeros(cov.shape[0]),
        cov=cov
    )

R_example = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0]
])