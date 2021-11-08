import os
import json
import copy
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
import meshio
import scipy.optimize
import matplotlib.pyplot as plt
from multiprocessing import Pool
from amortized_synthesis.utils import add_args, load_from_file, resample
from torch.utils.data import DataLoader
from amortized_synthesis.dataset import PathDataset, SoftRobotDataset
from amortized_synthesis.model import MLPModel, AutoEncoder
from amortized_synthesis.utils import path_chamfer_dis, normalize_path_tensor, optimizer_loss, rb_state_proj, \
    plot_mesh, rb_to_ratio, rb_smooth_regularizer, rb_sample_obs, rb_collision_penalty, rb_collision_detection, \
    load_rb_mesh_points_to_opt
from extruder_path.yaml_generator import path_save_to, get_material_para
from extruder_path.yaml_executor import get_exec_file_path
from soft_robot.data_generator import soft_robot_dispatcher
from repo_dir import get_repo_dir


def build_test_loader(opt):
    split = opt.train_val_test_split
    cum_split = [split[0], split[0] + split[1], split[0] + split[1] + split[2]]
    if opt.task == '3d-printing':
        data_opt = copy.deepcopy(opt)
        data_opt.model_name = 'encoder'
        test_dataset = PathDataset(data_opt, cum_split[1], cum_split[2])
    elif opt.task == 'soft-robot':
        test_dataset = SoftRobotDataset(opt, cum_split[1], cum_split[2])
    else:
        raise Exception()
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    return test_loader


def forward(model, data, opt):
    x, y, param = data
    x, y, param = x.to(opt.device), y.to(opt.device), param.to(opt.device)
    x, y, param = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0), torch.squeeze(param, dim=0)
    if opt.model_name == 'baseline':
        pred = model.forward(x)
    elif opt.model_name == 'encoder':
        pred = model.encoder.forward(x)
    else:
        raise Exception()
    pred = pred + param
    return pred, y


def path_vis(fib_path, ext_path, sim_fib_path, idx, output_dir):
    plt.figure(dpi=500)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(fib_path[:, 0], fib_path[:, 1], label='desired fiber path')
    plt.plot(ext_path[:, 0], ext_path[:, 1], label='extruder path (the solution)')
    if sim_fib_path is not None:
        plt.plot(sim_fib_path[:, 0], sim_fib_path[:, 1], label='simulated fiber path from the extruder path')
    plt.legend()
    plt.savefig(os.path.join(output_dir, '{:04d}.pdf'.format(idx)))
    plt.close()
    if sim_fib_path is not None:
        np.savez(os.path.join(output_dir, '{:04d}.npz'.format(idx)),
                 fib_path=fib_path, ext_path=ext_path, sim_fib_path=sim_fib_path)
    else:
        np.savez(os.path.join(output_dir, '{:04d}.npz'.format(idx)),
                 fib_path=fib_path, ext_path=ext_path)


def _pt_fun(ext_path, model, fib_path, opt, need_grad, pred_only=False):
    model.eval()
    model.zero_grad()
    ext_path = torch.tensor(ext_path, dtype=torch.float, requires_grad=True, device=opt.device)
    assert ext_path.shape[0] % 2 == 0
    ext_path_2d = ext_path.view(ext_path.shape[0] // 2, 2)
    ext_path_2d_flipped = ext_path_2d.flip(0)
    x = []
    norm_params = []
    for idx in range(ext_path_2d.shape[0]):
        segment_left = resample(ext_path_2d_flipped[ext_path_2d.shape[0]-idx-1:],
                                opt.pt_input_step_siz, opt.pt_input_half_len).flip(0)
        segment_right = resample(ext_path_2d[idx:], opt.pt_input_step_siz, opt.pt_input_half_len)
        segment_x = torch.cat((segment_left[:-1], segment_right), dim=0)
        segment_x, norm_param = normalize_path_tensor(segment_x)
        x.append(segment_x.reshape(1, -1))
        norm_params.append(norm_param.view(1, -1))
    x = torch.cat(x, dim=0)
    norm_params = torch.cat(norm_params, dim=0)
    norm_y = model.forward(x)
    y = norm_y + norm_params
    if pred_only:
        return y.cpu().detach().numpy()
    loss = optimizer_loss(path_2d=ext_path_2d, y=y, resampled_target=fib_path, opt=opt)
    if not need_grad:
        return loss.item()
    else:
        ext_path.retain_grad()
        loss.backward()
        return ext_path.grad.cpu().detach().numpy()


def fun(ext_path, model, fib_path, opt):
    if opt.task == '3d-printing':
        loss = _pt_fun(ext_path, model, fib_path, opt, need_grad=False)
    elif opt.task == 'soft-robot':
        loss = _rb_fun(ext_path, model, fib_path, opt, need_grad=False)
    else:
        raise Exception()
    return loss


def jac(ext_path, model, fib_path, opt):
    if opt.task == '3d-printing':
        grad = _pt_fun(ext_path, model, fib_path, opt, need_grad=True)
    elif opt.task == 'soft-robot':
        grad = _rb_fun(ext_path, model, fib_path, opt, need_grad=True)
    else:
        raise Exception()
    return grad


def pt_optimize(arg):
    fib_path, opt = arg
    fib_path_tensor = torch.tensor(fib_path, device=opt.device)
    model = MLPModel(opt)
    state_dict = torch.load(opt.trained_model_path, map_location=torch.device(opt.device))
    model.load_state_dict(state_dict)
    model = model.to(opt.device)
    solution = fib_path.flatten()
    res = scipy.optimize.minimize(fun=fun, jac=jac, x0=solution, method='BFGS',
                                  args=(model, fib_path_tensor, opt),
                                  options={'maxiter': opt.bfgs_max_iter, 'gtol': opt.bfgs_gtol})
    solution = res.x
    pred_fib_path = _pt_fun(ext_path=solution, model=model, fib_path=fib_path_tensor,
                            opt=opt, need_grad=False, pred_only=True)
    return np.reshape(solution, (-1, 2)), pred_fib_path


def pt_optimizer_main(test_loader, opt):
    ext_paths, fib_paths, trans = [], [], []
    args = []
    for data in tqdm(test_loader):
        _, y, _ = data
        y = torch.squeeze(y, dim=0).detach().cpu().numpy()
        fib_paths.append(y)
        args.append((y, opt))
    if opt.single_process_optimization:
        solutions = []
        inference_time_list = []
        with open(os.path.join(opt.save_to, 'time_log'), 'w') as time_log:
            for arg in tqdm(args):
                starting_time = time.time()
                solutions.append(pt_optimize(arg))
                inference_time_list.append(time.time() - starting_time)
                time_log.write("{}\n".format(time.time() - starting_time))
                time_log.flush()
            time_log.write("avg: {}, std: {}\n".format(np.mean(inference_time_list), np.std(inference_time_list)))
    else:
        with Pool(opt.pool_size) as p:
            solutions = list(tqdm(p.imap(pt_optimize, args), total=len(args)))
    output_dir = os.path.join(opt.save_to, 'optimization')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx in range(len(solutions)):
        solution, pred_fib_path = solutions[idx]
        fib_path = fib_paths[idx]
        plt.figure(dpi=500)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(fib_path[:, 0], fib_path[:, 1], label='desired fiber path')
        plt.plot(solution[:, 0], solution[:, 1], label='extruder path (the solution)')
        plt.plot(pred_fib_path[:, 0], pred_fib_path[:, 1], label='nn predicted fiber path from the extruder path')
        plt.legend()
        plt.savefig(os.path.join(output_dir, '{:04d}.pdf'.format(idx)))
        plt.close()
        np.savez(os.path.join(output_dir, '{:04d}.npz'.format(idx)), ext_path=solution, pred_fib_path=pred_fib_path)
        trans.append(copy.deepcopy(solution[0:1]))
        ext_path = solution - solution[0:1]
        ext_paths.append(ext_path)
    return ext_paths, fib_paths, trans


def pt_evaluate(test_loader, opt, model):
    if opt.model_name == 'optimizer':
        assert model is None
        ext_paths, fib_paths, trans = pt_optimizer_main(test_loader=test_loader, opt=opt)
    else:
        assert model is not None
        model.eval()
        init_forward(model, opt)
        ext_paths, fib_paths, trans = [], [], []
        inference_time_list = []
        with open(os.path.join(opt.save_to, 'time_log'), 'w') as time_log:
            for data in tqdm(test_loader):
                starting_time = time.time()
                ext_path, fib_path = forward(model, data, opt)
                inference_time_list.append(time.time() - starting_time)
                time_log.write("{}\n".format(time.time() - starting_time))
                time_log.flush()
                ext_path = ext_path.detach().cpu().numpy()
                fib_path = fib_path.detach().cpu().numpy()
                trans.append(copy.deepcopy(ext_path[0:1]))
                ext_path = ext_path - ext_path[0:1]
                ext_paths.append(ext_path)
                fib_paths.append(fib_path)
            time_log.write("avg: {}, std: {}\n".format(np.mean(inference_time_list), np.std(inference_time_list)))
    output_dir = os.path.join(opt.save_to, 'simulation')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    rst_paths = None
    if not opt.inference_only:
        exec_file_path = get_exec_file_path()
        tasks, rst_paths = [], []
        with open(os.path.join(output_dir, 'list'), 'w') as log:
            for idx in range(len(ext_paths)):
                yml_path = os.path.join(output_dir, '{:04d}.yml'.format(idx))
                rst_path = os.path.join(output_dir, '{:04d}.txt'.format(idx))
                log.write(yml_path + ',' + rst_path + '\n')
                path_save_to(ext_paths[idx], yml_path, override=get_material_para(opt.pt_material))
                rst_paths.append(rst_path)
                cmd_line = '{} \'{}\' \'{}\' VIS_OFF'.format(exec_file_path, yml_path, rst_path)
                tasks.append(cmd_line)
        with Pool(opt.pool_size) as p:
            list(tqdm(p.imap_unordered(os.system, tasks), total=len(tasks)))
    with open(os.path.join(opt.save_to, 'results'), 'w') as rst:
        good_cnt, bad_cnt = 0., 0.
        dis_sf_list, dis_fs_list = [], []
        for idx in tqdm(range(len(ext_paths))):
            rope_final_path = None
            if not opt.inference_only:
                printer_path, rope_final_path, rope_init_path = load_from_file(rst_paths[idx])
            if rope_final_path is None:
                bad_cnt += 1
                path_vis(fib_path=fib_paths[idx], ext_path=ext_paths[idx] + trans[idx], sim_fib_path=None, idx=idx,
                         output_dir=output_dir)
                continue
            sim_fib_path = rope_final_path + trans[idx]
            path_vis(fib_path=fib_paths[idx], ext_path=ext_paths[idx] + trans[idx], sim_fib_path=sim_fib_path, idx=idx,
                     output_dir=output_dir)
            dis_sf, dis_fs = path_chamfer_dis(sim_fib_path, fib_paths[idx])
            good_cnt += 1
            dis_sf_list.append(dis_sf)
            dis_fs_list.append(dis_fs)
            rst.write('{}, {}\n'.format(dis_sf, dis_fs))
        if not opt.inference_only:
            rst.write('good_cnt: {}\n'.format(good_cnt))
            rst.write('bad_cnt: {}\n'.format(bad_cnt))
            rst.write('dis_sf_avg: {}, dis_sf_std: {}\n'.format(np.mean(dis_sf_list), np.std(dis_sf_list)))
            rst.write('dis_fs_avg: {}, dis_fs_std: {}\n'.format(np.mean(dis_fs_list), np.std(dis_fs_list)))
            dis_list = (np.array(dis_sf_list) + np.array(dis_fs_list)) / 2.
            rst.write('dis_avg: {}, dis_std: {}\n'.format(np.mean(dis_list), np.std(dis_list)))


def _rb_fun(control_vector, model, target, opt, need_grad):
    target_vector, obstacle = target
    model.eval()
    model.zero_grad()
    control_vector = torch.tensor(control_vector, dtype=torch.float, requires_grad=True, device=opt.device)
    control_ratio_vector = rb_to_ratio(control_vector).unsqueeze(0)
    state_vector = model.forward(control_ratio_vector)
    loss = torch.mean((rb_state_proj(state_vector[0]) - target_vector) ** 2) + \
        opt.smooth_weight * rb_smooth_regularizer(control_ratio_vector, opt)
    if opt.rb_has_obstacle:
        assert obstacle is not None
        loss = loss + opt.rb_obstacle_penalty_weight * rb_collision_penalty(
            location_vectors=state_vector + opt.rb_mesh_points_tensor.unsqueeze(0),
            obstacles=obstacle.unsqueeze(0), opt=opt)
    if not need_grad:
        return loss.item()
    else:
        control_vector.retain_grad()
        loss.backward()
        return control_vector.grad.cpu().detach().numpy()


def rb_optimize(arg):
    target_vector, obstacle, opt = arg
    target_tensor = torch.tensor(target_vector, device=opt.device)
    opt_model = copy.deepcopy(opt)
    opt_model.model_name = 'decoder'
    model = MLPModel(opt_model)
    state_dict = torch.load(opt.trained_model_path, map_location=torch.device(opt.device))
    model.load_state_dict(state_dict)
    model = model.to(opt.device)
    res = scipy.optimize.minimize(fun=fun, jac=jac, x0=np.zeros(opt.rb_control_siz), method='BFGS',
                                  args=(model, (target_tensor, obstacle), opt),
                                  options={'maxiter': opt.bfgs_max_iter, 'gtol': opt.bfgs_gtol})
    solution = res.x
    return rb_to_ratio(solution)


def rb_optimizer_main(test_loader, obstacles, opt):
    target_vectors = []
    for _, state_vector in tqdm(test_loader):
        state_vector = torch.reshape(state_vector, (state_vector.size(0), -1))
        target_vector = rb_state_proj(state_vector).detach().cpu().numpy()
        target_vectors.append(target_vector)
    target_vectors = np.concatenate(target_vectors)
    if opt.rb_has_obstacle:
        args = [(target_vectors[idx], obstacles[idx], opt) for idx in range(len(target_vectors))]
    else:
        args = [(target_vectors[idx], None, opt) for idx in range(len(target_vectors))]
    if opt.single_process_optimization:
        inference_time_list = []
        pred_control_vectors = []
        with open(os.path.join(opt.save_to, 'time_log'), 'w') as time_log:
            for arg in tqdm(args):
                starting_time = time.time()
                pred_control_vectors.append(rb_optimize(arg))
                inference_time_list.append(time.time() - starting_time)
                time_log.write("{}\n".format(time.time() - starting_time))
                time_log.flush()
            time_log.write("avg: {}, std: {}\n".format(np.mean(inference_time_list), np.std(inference_time_list)))
    else:
        with Pool(opt.pool_size) as p:
            pred_control_vectors = list(tqdm(p.imap(rb_optimize, args), total=len(args)))
    pred_control_vectors = np.array(pred_control_vectors)
    return pred_control_vectors, target_vectors


def init_forward(model, opt):
    if opt.task == '3d-printing':
        rand_vector = torch.rand(1, (opt.pt_input_half_len * 2 + 1) * 2, dtype=torch.float32)
    elif opt.task == 'soft-robot':
        rand_vector = torch.rand(1, opt.rb_encoder_input_siz, dtype=torch.float32)
    else:
        raise Exception()
    rand_vector = rand_vector.to(opt.device)
    if opt.model_name == 'baseline':
        model.forward(rand_vector)
    elif opt.model_name == 'encoder':
        model.encoder.forward(rand_vector)
    else:
        raise Exception()


def rb_evaluate(test_loader, opt, model):
    if opt.rb_has_obstacle:
        np.random.seed(0)
        obstacles = rb_sample_obs(size=opt.train_val_test_split[2], opt=opt)
    else:
        obstacles = None
    if opt.model_name == 'optimizer':
        assert model is None
        pred_control_vectors, target_vectors = rb_optimizer_main(test_loader=test_loader, obstacles=obstacles, opt=opt)
    elif opt.model_name in ['baseline', 'encoder']:
        assert model is not None
        model.eval()
        init_forward(model, opt)
        pred_control_vectors, target_vectors = [], []
        inference_time_list = []
        sample_cnt = 0
        with open(os.path.join(opt.save_to, 'time_log'), 'w') as time_log:
            for _, state_vector in tqdm(test_loader):
                starting_time = time.time()
                state_vector = state_vector.to(opt.device)
                state_vector = torch.reshape(state_vector, (state_vector.size(0), -1))
                if opt.rb_has_obstacle:
                    obstacles_batch = obstacles[sample_cnt:sample_cnt + len(state_vector)]
                    inputs = torch.cat((rb_state_proj(state_vector), obstacles_batch), dim=1)
                else:
                    inputs = rb_state_proj(state_vector)
                if opt.model_name == 'baseline':
                    pred_control_vector = model.forward(inputs)
                elif opt.model_name == 'encoder':
                    pred_control_vector = model.encoder.forward(inputs)
                else:
                    raise Exception()
                inference_time_list.append(time.time() - starting_time)
                time_log.write("{}\n".format(time.time() - starting_time))
                time_log.flush()
                pred_control_vectors.append(pred_control_vector.detach().cpu().numpy())
                target_vectors.append(rb_state_proj(state_vector).detach().cpu().numpy())
                sample_cnt += len(state_vector)
            time_log.write("avg: {}, std: {}\n".format(np.mean(inference_time_list), np.std(inference_time_list)))
        pred_control_vectors = np.concatenate(pred_control_vectors)
        target_vectors = np.concatenate(target_vectors)
    else:
        raise Exception()
    if opt.inference_only:
        return
    output_dir = os.path.join(opt.save_to, 'simulation')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    saving_paths = [os.path.join(output_dir, '{:04d}'.format(idx)) for idx in range(len(pred_control_vectors))]
    sim_state_vectors = soft_robot_dispatcher(sources=pred_control_vectors, pool_size=1, saving_paths=saving_paths)
    sim_target_vectors = rb_state_proj(np.reshape(sim_state_vectors, [sim_state_vectors.shape[0], -1]))
    with open(os.path.join(opt.save_to, 'results'), 'w') as rst:
        dis_list = []
        succ_dis_list = []
        for idx in tqdm(range(len(sim_target_vectors))):
            dis = np.sum((sim_target_vectors[idx] - target_vectors[idx]) ** 2) ** 0.5
            mesh = meshio.read(saving_paths[idx] + '000000.vtu')
            plt.figure(dpi=500)
            plt.gca().set_aspect('equal', adjustable='box')
            location_vector = mesh.points[:, :2] + mesh.point_data['u'][:, :2]
            if opt.rb_has_obstacle:
                obstacle = obstacles[idx].detach().cpu().numpy()
                plt.gca().add_patch(plt.Circle(obstacle, radius=opt.rb_obstacle_radius, color='r', fill=False))
                plt.gca().add_patch(plt.Circle(obstacle, radius=opt.rb_obstacle_radius + opt.rb_obstacle_penalty_radius,
                                    color='b', fill=False))
            plot_mesh(location_vector, mesh.cells_dict['triangle'])
            target_abs = rb_state_proj((mesh.points[:, :2].flatten())) + target_vectors[idx]
            plt.plot(target_abs[0], target_abs[1], marker=(5, 1))
            plt.savefig(saving_paths[idx] + '.pdf')
            plt.close()
            if opt.rb_has_obstacle:
                np.savez(saving_paths[idx] + '.npz', target_vector=target_vectors[idx],
                         pred_control_vector=pred_control_vectors[idx], sim_state_vector=sim_state_vectors[idx],
                         obstacle=obstacle)
            else:
                np.savez(saving_paths[idx] + '.npz', target_vector=target_vectors[idx],
                         pred_control_vector=pred_control_vectors[idx], sim_state_vector=sim_state_vectors[idx])
            if opt.rb_has_obstacle:
                has_collision = rb_collision_detection(location_vectors=location_vector.flatten()[np.newaxis],
                                                       obstacles=obstacle[np.newaxis], opt=opt)
                if not has_collision.any():
                    succ_dis_list.append(dis)
                    rst.write('{}, success\n'.format(dis))
                else:
                    rst.write('{}, fail\n'.format(dis))
            else:
                rst.write('{}\n'.format(dis))
            dis_list.append(dis)
        rst.write('dis_avg: {}, dis_std: {}\n'.format(np.mean(dis_list), np.std(dis_list)))
        if opt.rb_has_obstacle:
            rst.write('succ_cnt: {}\n'.format(len(succ_dis_list)))
            rst.write('succ_dis_avg: {}, succ_dis_std: {}\n'.format(np.mean(succ_dis_list), np.std(succ_dis_list)))


def evaluate(test_loader, opt, model=None):
    if opt.task == '3d-printing':
        pt_evaluate(test_loader=test_loader, opt=opt, model=model)
    elif opt.task == 'soft-robot':
        rb_evaluate(test_loader=test_loader, opt=opt, model=model)
    else:
        raise Exception()


def main():
    parser = argparse.ArgumentParser()
    add_args(parser, testing=True)
    opt = parser.parse_args()
    if opt.task == '3d-printing':
        if opt.pt_material == 'carbon-fiber':
            opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'carbon-fiber')
        elif opt.pt_material == 'kevlar':
            opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'kevlar')
        else:
            raise Exception()
        assert opt.batch_size == 1
        if opt.model_name in ['decoder', 'baseline', 'encoder']:
            opt.smooth_weight = 0.
    elif opt.task == 'soft-robot':
        opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'soft-robot')
        opt.rb_control_siz = 40
        opt.rb_state_siz = 206
        opt.rb_encoder_input_siz = 4 if opt.rb_has_obstacle else 2
        if opt.rb_has_obstacle:
            assert opt.model_name in ['baseline', 'encoder', 'optimizer']
        if opt.model_name in ['baseline', 'encoder']:
            opt.rb_obstacle_penalty_weight = 0.
    else:
        raise Exception()
    if opt.task == '3d-printing':
        if opt.pt_test_on_shapes:
            opt.dataset_dir = ['square', 'star', 'h', 'square_scale_2', 'star_scale_2', 'h_scale_2']
            opt.train_val_test_split = [0, 0, len(opt.dataset_dir)]
    if opt.device == 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not torch.cuda.is_available() and opt.device == 'cuda:0':
        print("You do not have a CUDA device -- running on cpu.")
        opt.device = 'cpu'
    if not os.path.exists(opt.save_to):
        os.makedirs(opt.save_to)
    with open(os.path.join(opt.save_to, 'opt.json'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    if opt.task == 'soft-robot':
        load_rb_mesh_points_to_opt(opt)
    test_loader = build_test_loader(opt)
    if opt.model_name == 'encoder':
        model = AutoEncoder(opt)
    elif opt.model_name == 'baseline':
        model = MLPModel(opt)
    elif opt.model_name == 'optimizer':
        evaluate(test_loader=test_loader, opt=opt)
        return
    else:
        raise Exception()
    model = model.to(opt.device)
    state_dict = torch.load(opt.trained_model_path, map_location=torch.device(opt.device))
    model.load_state_dict(state_dict)
    evaluate(test_loader=test_loader, opt=opt, model=model)


if __name__ == '__main__':
    main()
