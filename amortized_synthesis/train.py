import os
import json
import numpy as np
from tqdm import tqdm
import torch
import argparse
from amortized_synthesis.utils import add_args, AverageMeter, stable_smooth_regularizer, rb_state_proj, \
    rb_smooth_regularizer, rb_sample_obs, rb_collision_penalty, rb_collision_detection, load_rb_mesh_points_to_opt
from amortized_synthesis.dataset import PathDataset, SoftRobotDataset
from amortized_synthesis.model import MLPModel, AutoEncoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from repo_dir import get_repo_dir


def build_dataloaders(opt):
    split = opt.train_val_test_split
    cum_split = [split[0], split[0] + split[1], split[0] + split[1] + split[2]]
    if opt.task == '3d-printing':
        dataset_builder = PathDataset
    elif opt.task == 'soft-robot':
        dataset_builder = SoftRobotDataset
    else:
        raise Exception()
    train_dataset = dataset_builder(opt, 0, cum_split[0])
    val_dataset = dataset_builder(opt, cum_split[0], cum_split[1])
    test_dataset = dataset_builder(opt, cum_split[1], cum_split[2])
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    else:
        raise Exception()
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    else:
        val_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def pt_forward(model, data, opt):
    x, y, param = data
    x, y, param = x.to(opt.device), y.to(opt.device), param.to(opt.device)
    x, y, param = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0), torch.squeeze(param, dim=0)
    if opt.model_name in ['baseline', 'decoder']:
        pred = model.forward(x)
        if torch.isnan(pred).any():
            print('nan in pred')
            quit()
        loss = torch.sum((pred - y) ** 2)
        if opt.model_name == 'baseline':
            loss = loss + opt.smooth_weight * stable_smooth_regularizer(pred + param, opt)
        return loss, len(x)
    elif opt.model_name in ['encoder']:
        ext_path, fib_path = model.forward(x, param)
        loss = torch.sum((fib_path - y) ** 2) + \
            opt.smooth_weight * stable_smooth_regularizer(ext_path, opt)
        return loss, len(x)
    else:
        raise Exception()


def sample_avoid(location_vectors, opt):
    assert len(location_vectors.shape) == 2 and location_vectors.shape[0] == 1 and \
        location_vectors.shape[1] == opt.rb_state_siz
    while True:
        obstacles = rb_sample_obs(size=1, opt=opt, need_tensor=False)
        has_collision = rb_collision_detection(location_vectors=location_vectors, obstacles=obstacles, opt=opt,
                                               for_baseline_training=True)
        if not has_collision.any():
            break
    obstacles = torch.tensor(obstacles, dtype=torch.float32, device=opt.device)
    return obstacles


def rb_forward(model, data, opt):
    control, state = data
    control, state = control.to(opt.device), state.to(opt.device)
    state = torch.reshape(state, (state.size(0), -1))
    if opt.model_name in ['decoder']:
        pred_state = model.forward(control)
        loss = torch.sum(torch.mean((pred_state - state) ** 2, dim=1))
    elif opt.model_name in ['baseline']:
        if opt.rb_has_obstacle:
            obstacles = []
            for state_vector in state:
                location_vector = state_vector.detach().cpu().numpy() + opt.rb_mesh_points
                obstacle = sample_avoid(location_vectors=location_vector[np.newaxis], opt=opt)
                obstacles.append(obstacle)
            obstacles = torch.cat(obstacles, dim=0)
            inputs = torch.cat((rb_state_proj(state), obstacles), dim=1)
            pred_control = model.forward(inputs)
        else:
            pred_control = model.forward(rb_state_proj(state))
        loss = torch.sum(torch.mean((pred_control - control) ** 2, dim=1)) + \
            opt.smooth_weight * rb_smooth_regularizer(pred_control, opt)
    elif opt.model_name in ['encoder']:
        if opt.rb_has_obstacle:
            obstacles = rb_sample_obs(size=len(control), opt=opt)
            inputs = torch.cat((rb_state_proj(state), obstacles), dim=1)
            pred_control, pred_state = model.forward(inputs)
        else:
            obstacles = None
            pred_control, pred_state = model.forward(rb_state_proj(state))
        loss = torch.sum(torch.mean((rb_state_proj(pred_state) - rb_state_proj(state)) ** 2, dim=1)) + \
            opt.smooth_weight * rb_smooth_regularizer(pred_control, opt)
        if opt.rb_has_obstacle:
            assert obstacles is not None
            loss = loss + opt.rb_obstacle_penalty_weight * rb_collision_penalty(
                location_vectors=pred_state + opt.rb_mesh_points_tensor.unsqueeze(0), obstacles=obstacles, opt=opt)
    else:
        raise Exception()
    return loss, len(control)


def forward(model, data, opt):
    if opt.task == '3d-printing':
        return pt_forward(model, data, opt)
    elif opt.task == 'soft-robot':
        return rb_forward(model, data, opt)
    else:
        raise Exception()


def train_val(model, train_loader, val_loader, writer, opt):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.learning_rate_decay)
    best_val_loss = None
    it_cnt = 0
    for epoch_cnt in range(opt.num_epochs):
        model.train()
        train_losses = AverageMeter()
        for data in tqdm(train_loader):
            loss, cnt = forward(model, data, opt)
            train_losses.update(loss, n=cnt)
            optimizer.zero_grad()
            loss.backward()
            if opt.model_name == 'encoder':
                model.decoder.zero_grad()
            optimizer.step()
            writer.add_scalar('Loss/train_it', loss, it_cnt + 1)
            it_cnt += 1
        lr_scheduler.step()
        writer.add_scalar('Loss/train', train_losses.avg, epoch_cnt + 1)

        if val_loader is None:
            continue
        model.eval()
        val_losses = AverageMeter()
        for data in tqdm(val_loader):
            loss, cnt = forward(model, data, opt)
            val_losses.update(loss, n=cnt)
        writer.add_scalar('Loss/val', val_losses.avg, epoch_cnt + 1)

        torch.save(model.state_dict(), os.path.join(opt.save_to, '{}.pt'.format(epoch_cnt + 1)))
        if best_val_loss is None or val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            torch.save(model.state_dict(), os.path.join(opt.save_to, 'best.pt'))


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    opt = parser.parse_args()
    if opt.task == '3d-printing':
        if opt.pt_material == 'carbon-fiber':
            opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'carbon-fiber')
        elif opt.pt_material == 'kevlar':
            opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'kevlar')
        else:
            raise Exception()
        assert opt.batch_size == 1
        if opt.model_name == 'decoder':
            opt.smooth_weight = 0.
    elif opt.task == 'soft-robot':
        opt.dataset_dir = os.path.join(get_repo_dir(), 'data', 'soft-robot')
        opt.rb_control_siz = 40
        opt.rb_state_siz = 206
        opt.rb_encoder_input_siz = 4 if opt.rb_has_obstacle else 2
        if opt.rb_has_obstacle:
            assert opt.model_name in ['baseline', 'encoder']
        if opt.model_name in ['decoder', 'baseline']:
            opt.rb_obstacle_penalty_weight = 0.
    else:
        raise Exception()
    if not torch.cuda.is_available() and opt.device == 'cuda:0':
        print("You do not have a CUDA device -- running on cpu.")
        opt.device = 'cpu'
    assert opt.model_name != 'optimizer'
    if not os.path.exists(opt.save_to):
        os.makedirs(opt.save_to)
    with open(os.path.join(opt.save_to, 'opt.json'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    if opt.task == 'soft-robot':
        load_rb_mesh_points_to_opt(opt)
    train_loader, val_loader, test_loader = build_dataloaders(opt)
    if opt.model_name == 'encoder':
        model = AutoEncoder(opt)
    elif opt.model_name in ['baseline', 'decoder']:
        model = MLPModel(opt)
    else:
        raise Exception()
    model = model.to(opt.device)
    writer = SummaryWriter(log_dir=opt.save_to)
    train_val(model=model, train_loader=train_loader, val_loader=val_loader, writer=writer, opt=opt)
    writer.close()


if __name__ == '__main__':
    main()
