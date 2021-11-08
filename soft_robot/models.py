import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from soft_robot.utils import batch_mat_vec


class RobotNetwork(nn.Module):
    def __init__(self, args, graph_info):
        super(RobotNetwork, self).__init__()
        self.args = args
        self.mat_list, self.joints, self.coo_diff, self.shapes = graph_info
        self.fcc1 = nn.Sequential(nn.Linear(args.input_size, args.input_size, bias=True), nn.Sigmoid())

        self.fcc2 = nn.Sequential(nn.Linear(self.shapes[0], self.shapes[1], bias=True))
        initialize_parameters([self.fcc1, self.fcc2], True)

    def get_angles(self, x):
        return 0.5 * np.pi + 0.5 * np.pi * 2 * (self.fcc1(x) - 0.5)

    def get_disp(self, x):
        angles = self.get_angles(x)
        lx_u, ly_u, rx_u, ry_u, bc_u = constrain(x, self.mat_list, self.coo_diff, self.joints, angles)
        int_u = self.fcc2(bc_u)
        return [lx_u, ly_u, rx_u, ry_u, bc_u], int_u

    def forward(self, x):
        [lx_u, ly_u, rx_u, ry_u, bc_u], int_u = self.get_disp(x)
        int_u = batch_mat_vec(self.mat_list[-1].transpose(0, 1), int_u)
        u = lx_u + ly_u + rx_u + ry_u + int_u
        return u


class RobotSolver(nn.Module):
    def __init__(self, args, graph_info):
        super(RobotSolver, self).__init__()
        self.args = args
        self.mat_list, self.joints, self.coo_diff, self.shapes = graph_info

        self.para_angles = torch.zeros(self.shapes[0] // 2) + 0.5 * np.pi
        self.para_disp = torch.zeros(self.shapes[1])

        self.para = torch.cat((self.para_angles, self.para_disp))
        self.para.requires_grad = True
        self.para = Parameter(self.para)

    def reset_parameters_network(self, source, robot_network):
        angles = robot_network.get_angles(source)
        self.para.data[:self.shapes[0] // 2] = angles.squeeze()
        _, int_u = robot_network.get_disp(source)
        self.para.data[self.shapes[0] // 2:] = int_u.squeeze()

    def reset_parameters_data(self, para_data):
        self.para.data = para_data

    def forward(self, x):
        self.para_angles = self.para[:self.shapes[0] // 2]
        self.para_disp = self.para[self.shapes[0] // 2:]
        lx_u, ly_u, rx_u, ry_u, bc_u = constrain(x, self.mat_list, self.coo_diff, self.joints,
                                                 self.para_angles.unsqueeze(0))
        int_u = batch_mat_vec(self.mat_list[-1].transpose(0, 1), self.para_disp.unsqueeze(0))
        u = lx_u + ly_u + rx_u + ry_u + int_u
        return u


def constrain(x, mat_list, coo_diff, joints, angles):
    half_size = x.shape[1] // 2
    ratio = 1 + (torch.sigmoid(x) - 0.5)

    new_rods_left = ratio[:, :half_size] * joints[0]
    new_rods_right = ratio[:, half_size:] * joints[1]

    new_rods_left = new_rods_left.unsqueeze(1).repeat(1, new_rods_left.shape[1], 1).triu()
    new_rods_right = new_rods_right.unsqueeze(1).repeat(1, new_rods_right.shape[1], 1).triu()

    cos_angle_left = torch.cos(angles[:, :half_size]).unsqueeze(2)
    sin_angle_left = torch.sin(angles[:, :half_size]).unsqueeze(2)
    cos_angle_right = torch.cos(angles[:, half_size:]).unsqueeze(2)
    sin_angle_right = torch.sin(angles[:, half_size:]).unsqueeze(2)

    lx_u = torch.matmul(new_rods_left, cos_angle_left).squeeze(2) + coo_diff[0]
    ly_u = torch.matmul(new_rods_left, sin_angle_left).squeeze(2) + coo_diff[1]
    rx_u = torch.matmul(new_rods_right, cos_angle_right).squeeze(2) + coo_diff[2]
    ry_u = torch.matmul(new_rods_right, sin_angle_right).squeeze(2) + coo_diff[3]
    bc_u = torch.cat([lx_u, ly_u, rx_u, ry_u], dim=1)

    lx_u = batch_mat_vec(mat_list[1].transpose(0, 1), lx_u)
    ly_u = batch_mat_vec(mat_list[2].transpose(0, 1), ly_u)
    rx_u = batch_mat_vec(mat_list[3].transpose(0, 1), rx_u)
    ry_u = batch_mat_vec(mat_list[4].transpose(0, 1), ry_u)

    return lx_u, ly_u, rx_u, ry_u, bc_u


def initialize_parameters(layers, bias_flag):
    for fcc in layers:
        for i, layer in enumerate(fcc):
            if i % 2 == 0:
                layer.weight.data[:] = 0
                if bias_flag:
                    layer.bias.data[:] = 0
