import os
import glob
from tqdm import tqdm
import numpy as np
import torch.utils.data
from amortized_synthesis.path import Path
from amortized_synthesis.utils import normalize_path, load_from_file, resample
from amortized_synthesis.path import get_path
from multiprocessing import Pool


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, opt, from_idx, to_idx):
        super().__init__()
        self.opt = opt
        if isinstance(self.opt.dataset_dir, str):
            txt_path_list = sorted(glob.glob(os.path.join(self.opt.dataset_dir, "*.txt")))[from_idx:to_idx]
        else:
            txt_path_list = self.opt.dataset_dir
        with Pool(8) as p:
            data = list(tqdm(p.imap(self.path_load_worker, txt_path_list), total=len(txt_path_list)))
        self.x_list = np.array([path[0] for path in data if path is not None])
        self.y_list = np.array([path[1] for path in data if path is not None])
        self.param_list = np.array([path[2] for path in data if path is not None])

    def centered_resample(self, path_x, path_y):
        x_list, y_list, param_list = [], [], []
        for point_idx in range(len(path_x.path)):
            time_steps = np.linspace(
                path_x.lengths[point_idx] - self.opt.pt_input_half_len * self.opt.pt_input_step_siz,
                path_x.lengths[point_idx] + self.opt.pt_input_half_len * self.opt.pt_input_step_siz,
                self.opt.pt_input_half_len * 2 + 1)
            seg_x = path_x.query(np.clip(time_steps, 0, path_x.lengths[-1]))
            y = path_y.path[point_idx]
            seg_x, y, norm_param = normalize_path(seg_x, y)
            x_list.append(seg_x)
            y_list.append(y)
            param_list.append(norm_param)
        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32), \
            np.array(param_list, dtype=np.float32)

    def path_load_worker(self, txt_path):
        if txt_path.endswith('.txt'):
            printer_path, rope_final_path, rope_init_path = load_from_file(txt_path, drop_last=2)
        else:
            assert self.opt.model_name == 'encoder'
            rope_init_path = None
            rope_final_path = get_path(txt_path)
        if rope_final_path is not None:
            if self.opt.model_name == 'decoder':
                ext_path = Path(rope_init_path)
                fib_path = Path(rope_final_path)
                return self.centered_resample(ext_path, fib_path)
            elif self.opt.model_name == 'baseline':
                ext_path = Path(rope_init_path)
                fib_path = Path(rope_final_path)
                return self.centered_resample(fib_path, ext_path)
            elif self.opt.model_name == 'encoder':
                resampled_fib_path = resample(rope_final_path, self.opt.pt_input_step_siz)
                path_len = len(resampled_fib_path)
                x_list, param_list = [], []
                for point_idx in range(path_len):
                    seg_x = []
                    for idx in range(point_idx - self.opt.pt_input_half_len,
                                     point_idx + self.opt.pt_input_half_len + 1):
                        seg_x.append(resampled_fib_path[max(min(idx, path_len - 1), 0)])
                    seg_x, norm_param = normalize_path(np.array(seg_x))
                    x_list.append(seg_x)
                    param_list.append(norm_param)
                return np.array(x_list, dtype=np.float32), np.array(resampled_fib_path, dtype=np.float32), \
                    np.array(param_list, dtype=np.float32)
            else:
                raise Exception()

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx], self.param_list[idx]

    def __len__(self):
        return len(self.x_list)


class SoftRobotDataset(torch.utils.data.Dataset):
    def __init__(self, opt, from_idx, to_idx):
        super().__init__()
        self.opt = opt
        input_file_paths = sorted(glob.glob(os.path.join(self.opt.dataset_dir, "input*")))
        self.controls = []
        self.states = []
        for input_file_path in input_file_paths:
            assert input_file_path.count('input') == 1
            output_file_path = input_file_path.replace('input', 'output')
            assert os.path.exists(output_file_path)
            self.controls.append(np.load(input_file_path))
            self.states.append(np.load(output_file_path))
        self.controls = np.concatenate(self.controls)[from_idx:to_idx].astype(np.float32)
        self.states = np.concatenate(self.states)[from_idx:to_idx].astype(np.float32)
        assert len(self.controls) == len(self.states)

    def __getitem__(self, idx):
        return self.controls[idx], self.states[idx]

    def __len__(self):
        return len(self.controls)
