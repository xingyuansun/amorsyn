import os
import copy
import torch
import scipy.special
import numpy as np
import datetime
import matplotlib.pyplot as plt
from soft_robot.data_generator import soft_robot_limit
from repo_dir import get_repo_dir


def add_args(parser, testing=False):
    parser.add_argument('--task', type=str, required=True, choices=['3d-printing', 'soft-robot'],
                        help='name of the task (3d-printing, soft-robot)')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['baseline', 'decoder', 'encoder', 'optimizer'],
                        help='name of the neural network')
    parser.add_argument('--mlp_hidden_sizes', type=int, nargs='+', default=[500, 200, 100, 50, 25],
                        help='hidden sizes of the model')
    parser.add_argument('--decoder_hidden_sizes', type=int, nargs='+', default=None,
                        help='hidden sizes of the decoder')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--train_val_test_split', type=int, nargs='+', required=True,
                        help='train/val/test split')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0'], default='cuda:0',
                        help='which device to use')
    parser.add_argument('--save_to', type=str,
                        default=os.path.join(get_repo_dir(), 'amortized_synthesis', 'runs',
                                             str(datetime.datetime.now())),
                        help='where to save the trained model')
    parser.add_argument('--smooth_weight', type=float, default=0.,
                        help='weight of the smooth regularizer (baseline/encoder/optimizer only)')
    parser.add_argument('--pt_material', type=str, choices=['carbon-fiber', 'kevlar'],
                        help='which material to use [carbon-fiber, kevlar] (3d printing)')
    parser.add_argument('--pt_input_half_len', type=int, default=30,
                        help='half length of the filter (3d printing)')
    parser.add_argument('--pt_input_step_siz', type=float, default=.03,
                        help='step size of the input (3d printing)')
    parser.add_argument('--rb_has_obstacle', action='store_true',
                        help='enable obstacle (soft robot)')
    parser.add_argument('--rb_obstacle_penalty_weight', type=float, default=0.,
                        help='weight of the obstacle penalty (encoder/optimizer only)')
    parser.add_argument('--rb_obstacle_radius', type=float, default=.9,
                        help='radius of the obstacle (soft robot)')
    parser.add_argument('--rb_obstacle_penalty_radius', type=float, default=.1,
                        help='penalty radius of the obstacle (soft robot)')
    parser.add_argument('--rb_obstacle_distribution_theta', type=float, default=np.pi/3.,
                        help='generate random obstacle from a sector with angle theta (soft robot)')
    parser.add_argument('--rb_obstacle_distribution_radius_1', type=float, default=4.,
                        help='sector radius 1 (soft robot)')
    parser.add_argument('--rb_obstacle_distribution_radius_2', type=float, default=5.,
                        help='sector radius 2 (soft robot)')
    if not testing:
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='learning rate')
        parser.add_argument('--learning_rate_decay', type=float, default=.98,
                            help='learning rate')
        parser.add_argument('--num_epochs', type=int, default=200,
                            help='number of epochs')
        parser.add_argument('--trained_decoder_path', type=str, default=None,
                            help='path to the trained decoder (for encoder)')
    else:
        parser.add_argument('--trained_model_path', type=str, required=True,
                            help='path to the trained model (for testing)')
        parser.add_argument('--pool_size', type=int, default=1,
                            help='number of processes (for testing)')
        parser.add_argument('--single_process_optimization', action='store_true',
                            help='single process optimization (for optimizer testing)')
        parser.add_argument('--inference_only', action='store_true',
                            help='inference only (for testing)')
        parser.add_argument('--bfgs_max_iter', type=int, default=None,
                            help='maximum number of iterations for BFGS')
        parser.add_argument('--bfgs_gtol', type=float, default=1e-5,
                            help='Gradient norm tolerance for BFGS')
        parser.add_argument('--pt_test_on_shapes', action='store_true',
                            help='test on shapes (for 3d-printing testing)')


def normalize_path(sample_a, sample_b=None):
    assert not isinstance(sample_a, torch.Tensor) and not isinstance(sample_b, torch.Tensor)
    offset = np.array(sample_a[len(sample_a) // 2])
    sample_a = sample_a - offset[np.newaxis, :]
    if sample_b is not None:
        sample_b = sample_b - offset
        return sample_a, sample_b, [offset[0], offset[1]]
    else:
        return sample_a, [offset[0], offset[1]]


def normalize_path_tensor(sample_a):
    assert isinstance(sample_a, torch.Tensor)
    offset = sample_a[sample_a.shape[0] // 2]
    sample_a = sample_a - offset.view(1, 2)
    return sample_a, offset


class AverageMeter(object):
    def __init__(self):
        self.sum = 0.
        self.count = 0
        self.avg = 0.

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        value = copy.deepcopy(value)
        self.sum += value
        self.count += n
        self.avg = self.sum / self.count


def path_chamfer_dis(a, b):
    dis = np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
    dis_ab = np.mean(np.min(dis, axis=1))
    dis_ba = np.mean(np.min(dis, axis=0))
    return dis_ab, dis_ba


def smooth_regularizer(path, input_step_siz):
    path_flipped = path.flip(0)
    regularizer = 0
    for idx in range(1, len(path) - 1):
        segment_left = resample(path_flipped[path.shape[0] - idx - 1:], input_step_siz, 1)
        segment_right = resample(path[idx:], input_step_siz, 1)
        regularizer = regularizer + torch.sum(((segment_left[1] + segment_right[1]) / 2. - path[idx]) ** 2)
    return regularizer


def stable_smooth_regularizer(path, opt, eps=1e-2):
    dis = torch.sum((path[1:] - path[:-1]) ** 2, dim=1, keepdim=True) ** 0.5 / opt.pt_input_step_siz
    trans = []
    cnt = 1
    for idx in range(1, len(path)):
        if dis[idx - 1] < opt.pt_input_step_siz * eps:
            cnt += 1
        else:
            row = torch.zeros(1, len(path))
            row[:, idx - cnt:idx] = 1. / cnt
            trans.append(row)
            cnt = 1
    row = torch.zeros(1, len(path))
    row[:, len(path) - cnt:len(path)] = 1. / cnt
    trans.append(row)
    trans = torch.cat(trans, dim=0).to(opt.device)
    path = torch.matmul(trans, path)

    dis = torch.sum((path[1:] - path[:-1]) ** 2, dim=1, keepdim=True) ** 0.5 / opt.pt_input_step_siz
    first_order = (path[1:] - path[:-1]) / (torch.clamp(dis, min=eps))
    second_order = (first_order[1:] - first_order[:-1]) / (torch.clamp((dis[1:] + dis[:-1]) / 2., min=eps))
    return torch.sum(second_order ** 2)


def optimizer_loss(path_2d, y, resampled_target, opt):
    loss, lap_reg, len_reg = torch.zeros(1).to(opt.device), torch.zeros(1).to(opt.device), torch.zeros(1).to(opt.device)

    for idx in range(1, path_2d.shape[0] - 1):
        lap_reg = lap_reg + l2_dis(path_2d[idx], (path_2d[idx - 1] + path_2d[idx + 1]) / 2., squared=True)

    for idx in range(1, path_2d.shape[0]):
        len_reg = len_reg + l2_dis(path_2d[idx - 1], path_2d[idx], squared=True)

    lap_reg = lap_reg / (len_reg ** 0.5)
    loss = loss + lap_reg * opt.smooth_weight

    target_len, y_len = path_len(resampled_target), path_len(y)
    target_idx, y_idx = 0, 0
    target_prop, y_prop = 0., 0.
    walked = 0.
    while True:
        if target_idx == len(resampled_target) - 1 or y_idx == len(y) - 1:
            break
        delta = min(
            l2_dis(resampled_target[target_idx], resampled_target[target_idx + 1]) * (1. - target_prop) / target_len,
            l2_dis(y[y_idx], y[y_idx + 1]) * (1. - y_prop) / y_len)
        target_origin = resampled_target[target_idx] + (
                resampled_target[target_idx + 1] - resampled_target[target_idx]) * target_prop
        y_origin = y[y_idx] + (y[y_idx + 1] - y[y_idx]) * y_prop
        target_d = (resampled_target[target_idx + 1] - resampled_target[target_idx]) / l2_dis(
            resampled_target[target_idx], resampled_target[target_idx + 1]) * delta * target_len
        y_d = (y[y_idx + 1] - y[y_idx]) / l2_dis(y[y_idx], y[y_idx + 1]) * delta * y_len
        loss = loss + l2_dis_segments(target_origin, target_d, y_origin, y_d) * delta
        target_prop = target_prop + delta * target_len / l2_dis(resampled_target[target_idx],
                                                                resampled_target[target_idx + 1])
        y_prop = y_prop + delta * y_len / l2_dis(y[y_idx], y[y_idx + 1])
        walked = walked + delta
        if target_prop > y_prop:
            target_prop = 0.
            target_idx = target_idx + 1
        else:
            y_prop = 0.
            y_idx = y_idx + 1

    return loss


def rb_state_proj(state):
    assert state.shape[-1] == 206
    if len(state.shape) == 2:
        return state[:, 122:124]
    elif len(state.shape) == 1:
        return state[122:124]
    else:
        raise Exception()


def rb_to_ratio(x):
    if isinstance(x, torch.Tensor):
        return (torch.sigmoid(x) * 2 - 1) * soft_robot_limit()
    else:
        return (scipy.special.expit(x) * 2 - 1) * soft_robot_limit()


def _rb_smooth_regularizer(vector):
    return torch.sum(torch.mean((vector[:, 1:-1] - (vector[:, 2:] + vector[:, :-2]) / 2.) ** 2, dim=1))


def rb_smooth_regularizer(control_vector, opt):
    assert control_vector.shape[1] == opt.rb_control_siz and opt.rb_control_siz % 2 == 0
    half_len = opt.rb_control_siz // 2
    return (_rb_smooth_regularizer(control_vector[:, :half_len]) +
            _rb_smooth_regularizer(control_vector[:, half_len:])) / 2.


def _dis_to_obs(location_vectors, obstacles, opt):
    assert opt.rb_has_obstacle
    assert location_vectors.shape[1] == opt.rb_state_siz and opt.rb_state_siz % 2 == 0 and obstacles.shape[1] == 2
    location_vectors = location_vectors.reshape((location_vectors.shape[0], location_vectors.shape[1] // 2, 2))
    obstacles = obstacles.reshape((obstacles.shape[0], 1, obstacles.shape[1]))
    dis_to_obs = ((location_vectors - obstacles) ** 2).sum(axis=2) ** 0.5
    return dis_to_obs


def rb_collision_penalty(location_vectors, obstacles, opt):
    dis_to_obs = _dis_to_obs(location_vectors, obstacles, opt)
    penalty = torch.sum(torch.mean(torch.clamp(-dis_to_obs + opt.rb_obstacle_penalty_radius
                                               + opt.rb_obstacle_radius, min=0.) ** 2, dim=1))
    return penalty


def plot_segment(a, b):
    plt.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=1)


def plot_mesh(points, cells):
    for x, y, z in cells:
        plot_segment(points[x], points[y])
        plot_segment(points[x], points[z])
        plot_segment(points[y], points[z])


def rb_sample_obs(size, opt, need_tensor=True):
    assert opt.rb_has_obstacle
    theta = np.random.uniform(low=(np.pi - opt.rb_obstacle_distribution_theta) / 2.,
                              high=(np.pi + opt.rb_obstacle_distribution_theta) / 2., size=size)
    radius = np.random.uniform(low=opt.rb_obstacle_distribution_radius_1,
                               high=opt.rb_obstacle_distribution_radius_2, size=size)
    x = np.cos(theta) * radius + opt.rb_mesh_points[2]
    y = np.sin(theta) * radius + opt.rb_mesh_points[3]
    obstacles = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    if need_tensor:
        obstacles = torch.tensor(obstacles, dtype=torch.float32, device=opt.device)
    return obstacles


def rb_collision_detection(location_vectors, obstacles, opt, for_baseline_training=False):
    dis_to_obs = _dis_to_obs(location_vectors, obstacles, opt)
    if for_baseline_training:
        has_collision = np.any(dis_to_obs < opt.rb_obstacle_radius + opt.rb_obstacle_penalty_radius, axis=1)
    else:
        has_collision = np.any(dis_to_obs < opt.rb_obstacle_radius, axis=1)
    return has_collision


def load_rb_mesh_points_to_opt(opt):
    opt.rb_mesh_points = np.load(os.path.join(get_repo_dir(), 'amortized_synthesis', 'rb_mesh_points.npy'))
    opt.rb_mesh_points_tensor = torch.tensor(opt.rb_mesh_points, dtype=torch.float32, device=opt.device)


def load_from_file(data_path, only_keep_xz=True, data_type=np.float32, drop_last=0):
    if not os.path.exists(data_path):
        return None, None, None
    with open(data_path) as f:
        data = f.read().split('\n')
    cnt = 0
    if data[cnt] == 'Time limit exceeded.':
        return None, None, None

    rst = []
    for i in range(3):
        path_len = int(data[cnt])
        path = np.zeros((path_len, 4))
        for i in range(path_len):
            cnt += 1
            path[i] = np.array(data[cnt].split(' '), dtype=data_type)
        cnt += 1
        if only_keep_xz:
            path = path[:, [0, 2]]
        rst.append(path)

    if drop_last > 0:
        return rst[0][:-drop_last], rst[1][:-drop_last], rst[2][:-drop_last]
    elif drop_last == 0:
        return rst[0], rst[1], rst[2]
    else:
        raise Exception()


def l2_dis(a, b, squared=False):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        answer = torch.sum((a - b) ** 2)
    elif isinstance(a, torch.Tensor):
        answer = torch.sum((a - torch.tensor(b, dtype=torch.double, device=a.device)) ** 2)
    elif isinstance(b, torch.Tensor):
        answer = torch.sum((b - torch.tensor(a, dtype=torch.double, device=b.device)) ** 2)
    else:
        answer = np.sum((a - b) ** 2)
    if squared:
        return answer
    else:
        return answer ** 0.5


def l2_dis_segments(origin1, d1, origin2, d2):
    x1, y1 = origin1
    u1, v1 = d1
    x2, y2 = origin2
    u2, v2 = d2
    a = (u1 - u2) ** 2 + (v1 - v2) ** 2
    b = 2. * (x1 - x2) * (u1 - u2) + 2. * (y1 - y2) * (v1 - v2)
    c = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return a / 3. + b / 2. + c


def path_len(path):
    length = 0.
    for idx in range(1, len(path)):
        length = length + l2_dis(path[idx - 1], path[idx])
    return length


def resample(path, step_siz, num_steps=None):
    needed_len = 0.
    len_to_idx = 0.
    idx = 0
    resampled_path = []
    while True:
        while idx + 1 < path.shape[0] and len_to_idx + l2_dis(path[idx], path[idx + 1]) + 1e-5 < needed_len:
            len_to_idx = len_to_idx + l2_dis(path[idx], path[idx + 1])
            idx += 1
        if idx + 1 >= path.shape[0]:
            if num_steps is None:
                break
            else:
                resampled_path.append(path[-1])
        else:
            resampled_path.append(path[idx] + (needed_len - len_to_idx) *
                                  (path[idx + 1] - path[idx]) / l2_dis(path[idx], path[idx + 1]))
        if isinstance(resampled_path[-1], torch.Tensor):
            resampled_path[-1] = resampled_path[-1].view(1, 2)
        else:
            resampled_path[-1] = np.reshape(resampled_path[-1], (1, 2))
        needed_len = needed_len + step_siz
        if num_steps is not None and len(resampled_path) >= num_steps + 1:
            break

    if isinstance(resampled_path[-1], torch.Tensor):
        return torch.cat(resampled_path, dim=0)
    else:
        return np.concatenate(resampled_path, axis=0)
