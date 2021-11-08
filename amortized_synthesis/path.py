import numpy as np
import math
import bisect
from argparse import Namespace


def l2_dis(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def namespace_from_dict(opt):
    if not isinstance(opt, dict):
        return opt
    for attr in opt.keys():
        opt[attr] = namespace_from_dict(opt[attr])
    ns = Namespace(**opt)
    return ns


class Path:
    def __init__(self, path):
        self.path = np.array(path)
        self.lengths = np.zeros(len(path))
        for idx in range(len(path)):
            if idx == 0:
                continue
            self.lengths[idx] = self.lengths[idx - 1] + l2_dis(self.path[idx - 1], self.path[idx])
        self.lengths = np.array(self.lengths)

    def query(self, t_list):
        answer = np.zeros([len(t_list), 2])
        for t_idx, t in enumerate(t_list):
            assert -1e-5 * self.lengths[-1] <= t <= (1 + 1e-5) * self.lengths[-1]
            if t < 1e-10 * self.lengths[-1]:
                idx = 0
            elif t > (1 - 1e-5) * self.lengths[-1]:
                idx = len(self.lengths) - 2
            else:
                idx = bisect.bisect_left(self.lengths, t) - 1
            if np.abs(self.lengths[idx + 1] - self.lengths[idx]) < 1e-7:
                answer[t_idx, 0] = (self.path[idx + 1, 0] + self.path[idx, 0]) / 2.
                answer[t_idx, 1] = (self.path[idx + 1, 1] + self.path[idx, 1]) / 2.
            else:
                dt = (t - self.lengths[idx]) / (self.lengths[idx + 1] - self.lengths[idx])
                answer[t_idx, 0] = dt * (self.path[idx + 1, 0] - self.path[idx, 0]) + self.path[idx, 0]
                answer[t_idx, 1] = dt * (self.path[idx + 1, 1] - self.path[idx, 1]) + self.path[idx, 1]
        return answer


def normalize_target_path(path, desired_len=4.35):
    path = np.array(path)
    path = path - path[0:1]
    length = 0.
    for idx in range(1, len(path)):
        length += l2_dis(path[idx - 1], path[idx])
    path = path * desired_len / length
    return path


def get_path_square():
    path = np.array([[0., 0.], [1.1, 0.], [1.1, 1.1], [0., 1.1], [0., 0.]])
    return normalize_target_path(path)


def get_path_star(n=5):
    path = []
    vertexes = [np.array([1., 1. / math.tan(math.radians(180. / n))]),
                np.array([0, 1. / math.tan(math.radians(180. / n)) + 1. / math.tan(math.radians(90. / n))])]
    for i in range(n):
        for j in range(len(vertexes)):
            path.append(vertexes[j].copy())
            rot_mat = np.array([[np.cos(math.radians(360. / n)), -np.sin(math.radians(360. / n))],
                                [np.sin(math.radians(360. / n)), np.cos(math.radians(360. / n))]])
            vertexes[j] = np.dot(rot_mat, vertexes[j].T).T
    path.append(path[0])
    return normalize_target_path(path)


def draw_arc(origin, theta_l, theta_r, radius, resolution=50):
    path = []
    origin = np.array(origin)
    for theta in np.linspace(theta_l, theta_r, resolution):
        path.append(origin + radius * np.array([math.cos(theta), math.sin(theta)]))
    return path


def get_path_h():
    path = []
    path.append([0, 3])
    path.extend(draw_arc([-5, 2.5], np.pi * 0.5, np.pi * 1.5, 0.5))
    path.extend(draw_arc([-1, 1.5], np.pi * 0.5, 0, 0.5))
    path.extend(draw_arc([-1, -1.5], 0, -np.pi * 0.5, 0.5))
    path.extend(draw_arc([-5, -2.5], np.pi * 0.5, np.pi * 1.5, 0.5))
    path.extend(draw_arc([5, -2.5], -np.pi * 0.5, np.pi * 0.5, 0.5))
    path.extend(draw_arc([1, -1.5], np.pi * 1.5, np.pi, 0.5))
    path.extend(draw_arc([1, 1.5], np.pi, np.pi * 0.5, 0.5))
    path.extend(draw_arc([5, 2.5], -np.pi * 0.5, np.pi * 0.5, 0.5))
    path.append([0, 3])
    return normalize_target_path(path)


def get_path_spiral(num, step_siz=.2):
    path = []
    x, y = 0, 0
    dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    path.append([x, y])
    for idx in range(num):
        x, y = x + dirs[idx % 4][0] * (idx // 2 + 1) * step_siz, y + dirs[idx % 4][1] * (idx // 2 + 1) * step_siz
        path.append([x, y])
    path = np.array(path, dtype=float)
    path = path - path[0:1]
    return path


def get_path_zigzag(num, step_siz=.12):
    path = []
    for idx in range(num):
        left = [idx * step_siz, 0.]
        right = [idx * step_siz, (num - 1) * step_siz]
        if idx % 2 == 0:
            path.extend([left, right])
        else:
            path.extend([right, left])
    path = np.array(path, dtype=float)
    path = path - path[0:1]
    return path


def get_path_intersect(r):
    path = []
    center_1 = np.array([2. * r, 0.])
    center_2 = np.array([2. * r, r])
    for theta in np.linspace(1. * np.pi, 0.5 * np.pi, 45):
        path.append(center_1 + 2. * r * np.array([np.cos(theta), np.sin(theta)]))
    for theta in np.linspace(0.5 * np.pi, -1.5 * np.pi, 360):
        path.append(center_2 + r * np.array([np.cos(theta), np.sin(theta)]))
    for theta in np.linspace(0.5 * np.pi, 0. * np.pi, 360):
        path.append(center_1 + 2. * r * np.array([np.cos(theta), np.sin(theta)]))
    path = np.array(path, dtype=float)
    # path = path - path[0:1]
    return path


def get_path_intersect2(d):
    path = [[0., 0.], [0., d], [d / 2., d / 2.], [-d / 2., d / 2.]]
    path = np.array(path, dtype=float)
    return path


def get_path(path_name):
    modifier, arg = None, None
    if '_' in path_name:
        path_name, modifier, arg = path_name.split('_')
    assert modifier in [None, 'scale', 'para']

    if path_name == 'square':
        shape = get_path_square()
    elif path_name == 'star':
        shape = get_path_star()
    elif path_name == 'h':
        shape = get_path_h()
    elif path_name == 'spiral':
        assert modifier == 'para'
        shape = get_path_spiral(int(arg))
    elif path_name == 'zigzag':
        assert modifier == 'para'
        shape = get_path_zigzag(int(arg))
    elif path_name == 'intersect':
        assert modifier == 'para'
        shape = get_path_intersect(float(arg))
    elif path_name == 'intersect2':
        assert modifier == 'para'
        shape = get_path_intersect2(float(arg))
    else:
        raise Exception()

    if modifier == 'scale':
        scale = float(arg)
        shape = shape * scale
    return shape

