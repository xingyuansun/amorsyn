import copy
import torch
import torch.nn
from amortized_synthesis.utils import normalize_path_tensor, rb_to_ratio, resample


class MLPModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        layers = []
        if opt.task == '3d-printing':
            ori_input_len = (self.opt.pt_input_half_len * 2 + 1) * 2
            layer_sizes = [ori_input_len] + self.opt.mlp_hidden_sizes + [2]
        elif opt.task == 'soft-robot':
            if opt.model_name in ['decoder']:
                layer_sizes = [self.opt.rb_control_siz] + self.opt.mlp_hidden_sizes + [self.opt.rb_state_siz]
            elif opt.model_name in ['baseline', 'encoder']:
                layer_sizes = [self.opt.rb_encoder_input_siz] + self.opt.mlp_hidden_sizes + [self.opt.rb_control_siz]
            else:
                raise Exception()
        else:
            raise Exception()
        for i in range(1, len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if self.opt.task == 'soft-robot' and self.opt.model_name in ['baseline', 'encoder']:
            return rb_to_ratio(self.model(x))
        else:
            return self.model(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder = MLPModel(self.opt)
        decoder_opt = copy.deepcopy(self.opt)
        decoder_opt.model_name = 'decoder'
        if decoder_opt.decoder_hidden_sizes is not None:
            decoder_opt.mlp_hidden_sizes = decoder_opt.decoder_hidden_sizes
        self.decoder = MLPModel(decoder_opt)
        if hasattr(self.opt, 'trained_decoder_path') and self.opt.trained_decoder_path is not None:
            state_dict = torch.load(self.opt.trained_decoder_path, map_location=torch.device(self.opt.device))
            self.decoder.load_state_dict(state_dict)

    def forward(self, x, param=None):
        if self.opt.task == '3d-printing':
            assert param is not None
            norm_ext_path = self.encoder(x)
            ext_path = norm_ext_path + param
            ext_path_flipped = ext_path.flip(0)

            x = []
            norm_params = []
            for idx in range(ext_path.shape[0]):
                segment_left = resample(ext_path_flipped[ext_path.shape[0] - idx - 1:], self.opt.pt_input_step_siz,
                                        self.opt.pt_input_half_len).flip(0)
                segment_right = resample(ext_path[idx:], self.opt.pt_input_step_siz, self.opt.pt_input_half_len)
                segment_x = torch.cat((segment_left[:-1], segment_right), dim=0)
                segment_x, norm_param = normalize_path_tensor(segment_x)
                x.append(segment_x.reshape(1, -1))
                norm_params.append(norm_param.view(1, -1))
            x = torch.cat(x, dim=0)
            norm_params = torch.cat(norm_params, dim=0)
            norm_y = self.decoder.forward(x)
            y = norm_y + norm_params
            return ext_path, y
        elif self.opt.task == 'soft-robot':
            assert param is None
            control_vec = self.encoder(x)
            state_vec = self.decoder(control_vec)
            return control_vec, state_vec
