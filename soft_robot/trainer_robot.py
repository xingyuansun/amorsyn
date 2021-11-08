import torch
from torch import optim
import numpy as np
import os
from soft_robot.utils import batch_mat_vec, boundary_flag_matrix, save_vtu, parse_vtu
from soft_robot.models import RobotNetwork, RobotSolver
from soft_robot.transformer import PoissonRobot
from tqdm import tqdm


class TrainerRobot:
    def __init__(self, args):
        self.args = args
        self.poisson = PoissonRobot(self.args)
        self.initialization()

    def loss_function(self, x_control, x_state, y_state=None):
        # y_state is dummy
        young_mod = 100
        poisson_ratio = 0.3
        shear_mod = young_mod / (2 * (1 + poisson_ratio))
        bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

        F00 = batch_mat_vec(self.F00, x_state) + 1
        F01 = batch_mat_vec(self.F01, x_state)
        F10 = batch_mat_vec(self.F10, x_state)
        F11 = batch_mat_vec(self.F11, x_state) + 1

        J = F00 * F11 - F01 * F10
        Jinv = J ** (-2 / 3)
        I1 = F00 * F00 + F01 * F01 + F10 * F10 + F11 * F11

        energy_density = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) + (bulk_mod / 2) * (J - 1) ** 2)

        loss = (energy_density * self.weight_area).sum()

        return loss

    def initialization(self):
        bc_btm, bc_lx, bc_ly, bc_rx, bc_ry = self.poisson.boundary_flags_list
        interior = np.ones(self.poisson.num_dofs)
        for bc in self.poisson.boundary_flags_list:
            interior -= bc

        bc_btm_mat = boundary_flag_matrix(bc_btm)
        bc_lx_mat = boundary_flag_matrix(bc_lx)
        bc_ly_mat = boundary_flag_matrix(bc_ly)
        bc_rx_mat = boundary_flag_matrix(bc_rx)
        bc_ry_mat = boundary_flag_matrix(bc_ry)
        int_mat = boundary_flag_matrix(interior)

        self.bc_l_mat = bc_lx_mat
        self.bc_r_mat = bc_rx_mat
        self.args.input_size = self.bc_l_mat.shape[0] + self.bc_r_mat.shape[0]

        interior_size = int_mat.sum()
        control_bc_size = bc_lx_mat.sum() + bc_ly_mat.sum() + bc_rx_mat.sum() + bc_ry_mat.sum()
        shapes = [int(control_bc_size), int(interior_size)]

        lx_0, ly_0, rx_0, ry_0 = 0, 0, self.poisson.width, 0
        lx = np.matmul(bc_lx_mat, self.poisson.coo_dof[:, 0])
        lx_new = np.diff(np.append(lx, lx_0))
        ly = np.matmul(bc_lx_mat, self.poisson.coo_dof[:, 1])
        ly_new = np.diff(np.append(ly, ly_0))
        rx = np.matmul(bc_rx_mat, self.poisson.coo_dof[:, 0])
        rx_new = np.diff(np.append(rx, rx_0))
        ry = np.matmul(bc_rx_mat, self.poisson.coo_dof[:, 1])
        ry_new = np.diff(np.append(ry, ry_0))
        lr = np.sqrt(lx_new ** 2 + ly_new ** 2)
        rr = np.sqrt(rx_new ** 2 + ry_new ** 2)
        joints = [torch.tensor(lr).float(), torch.tensor(rr).float()]
        coo_diff = [torch.tensor(lx_0 - lx).float(), torch.tensor(ly_0 - ly).float(),
                    torch.tensor(rx_0 - rx).float(), torch.tensor(ry_0 - ry).float()]

        F00, F01, F10, F11 = np.load(os.path.join(self.args.root_path, self.args.numpy_path, 'robot/F.npy'))

        self.F00 = torch.tensor(F00).float().to_sparse()
        self.F01 = torch.tensor(F01).float().to_sparse()
        self.F10 = torch.tensor(F10).float().to_sparse()
        self.F11 = torch.tensor(F11).float().to_sparse()
        self.weight_area = torch.tensor(self.poisson.compute_areas()).float()
        bc_btm_mat = torch.tensor(bc_btm_mat).float().to_sparse()
        bc_lx_mat = torch.tensor(bc_lx_mat).float().to_sparse()
        bc_ly_mat = torch.tensor(bc_ly_mat).float().to_sparse()
        bc_rx_mat = torch.tensor(bc_rx_mat).float().to_sparse()
        bc_ry_mat = torch.tensor(bc_ry_mat).float().to_sparse()
        int_mat = torch.tensor(int_mat).float().to_sparse()

        mat_list = [bc_btm_mat, bc_lx_mat, bc_ly_mat, bc_rx_mat, bc_ry_mat, int_mat]
        self.graph_info = [mat_list, joints, coo_diff, shapes]

    def forward_prediction(self, source, model=None, para_data=None):
        """Serves as ground truth computation
        Parameters
        ----------
        source: numpy array (n,)
        """
        source = np.expand_dims(source, axis=0)
        source = -np.log(1. / (source + 0.5) - 1.)
        source = torch.tensor(source, dtype=torch.float)
        solver = RobotSolver(self.args, self.graph_info)
        # Load pre-trained network for better convergence
        # model.load_state_dict(torch.load(self.args.root_path + '/' + self.args.model_path + '/robot/solver'))

        if model is not None:
            solver.reset_parameters_network(source, model)

        if para_data is not None:
            solver.reset_parameters_data(para_data)

        optimizer = optim.LBFGS(solver.parameters(), lr=1e-1, max_iter=20, history_size=100)
        max_epoch = 100000
        tol = 1e-20
        loss_pre, loss_crt = 0, 0
        for epoch in range(max_epoch):
            def closure():
                optimizer.zero_grad()
                solution = solver(source)
                loss = self.loss_function(source, solution)
                loss.backward()
                return loss

            solution = solver(source)
            loss = self.loss_function(source, solution)
            # print("Optimization for ground truth, loss is", loss.data.numpy())
            assert (not np.isnan(loss.data.numpy()))

            optimizer.step(closure)
            loss_pre = loss_crt
            loss_crt = loss.data.numpy()
            if (loss_pre - loss_crt) ** 2 < tol:
                break

        return solution[0].data.numpy(), solver.para.data

    def simulate(self, sources, saving_paths):
        """source -> solution
        """
        self.model = RobotNetwork(self.args, self.graph_info)
        self.model.load_state_dict(torch.load(os.path.join(self.args.root_path, self.args.model_path, 'robot/model_s')))
        displacements = []
        for idx, source in tqdm(list(enumerate(sources))):
            solution, _ = self.forward_prediction(source, model=self.model)
            pvd_path = saving_paths[idx] + '.pvd'
            vtu_path = saving_paths[idx] + '000000.vtu'
            save_vtu(attribute=solution, pde=self.poisson, saving_path=pvd_path)
            mesh = parse_vtu(vtu_path)
            displacements.append(mesh.point_data['u'][np.newaxis, :, :2])
        return np.concatenate(displacements)
