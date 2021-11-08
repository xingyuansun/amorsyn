import numpy as np
import numpy.linalg as npla
import scipy.special as spec
from extruder_path.external.elliptical_slice import elliptical_slice
import segment_intersection


def get_segments(points):
    num_points = points.shape[0]
    segments = np.zeros((4, num_points, 1))
    segments[:2, :, 0] = points.T
    segments[2:, :-1, 0] = points[1:, :].T
    segments[:2, -1, 0] = points[0, :]
    return segments


def log_like(Y, *args):
    YY = np.reshape(Y, (2, -1)).T
    S = get_segments(YY)
    has_intersection = segment_intersection.query(S[0, :, 0], S[1, :, 0], S[2, :, 0], S[3, :, 0], args[0])
    return -np.inf if has_intersection else 0


def random_jordan_curve(ls=0.1, N=1000, eps=1e-7):
    np.random.seed()
    theta = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    # find the covariance function
    K = np.exp(-2 * np.sin((theta[:, np.newaxis] - theta[np.newaxis, :]) / 2) ** 2 / ls ** 2) + 1e-6 * np.eye(N)
    cholK = npla.cholesky(K)

    Y = np.vstack([np.cos(theta), np.sin(theta)]).T
    Y = Y.T.ravel()
    for _ in range(1000):
        Y, cur_log_like = elliptical_slice(Y, np.kron(np.eye(2), cholK), log_like, pdf_params=(eps,))
    Y = np.reshape(Y, (2, -1)).T
    Z = 0.75 * (1 + spec.erf(Y / np.sqrt(2)))
    points = [(z[0], z[1]) for z in Z]
    points.append(points[0])  # loops back

    points = np.array(points)
    return points
