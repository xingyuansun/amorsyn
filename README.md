# Amortized Synthesis of Constrained Configurations Using a Differentiable Surrogate

This repository is the implementation of 
[Amortized Synthesis of Constrained Configurations Using a Differentiable Surrogate](https://arxiv.org/abs/2106.09019),
NeurIPS 2021, Spotlight. 
```
@InProceedings{amorsyn,
  title={{Amortized Synthesis of Constrained Configurations Using a Differentiable Surrogate}},
  author={Sun, Xingyuan and Xue, Tianju and Rusinkiewicz, Szymon and Adams, Ryan P},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

There are 3 folders: 
* `extruder_path` contains the code to generate synthetic data for case study extruder path planning,
* `soft_robot` contains the code to generate synthetic data for case study constrained soft robot inverse kinematics,
which is developed from [AmorFEA](https://github.com/tianjuxue/AmorFEA),
* `amortized_synthesis` contains the code of our amortized synthesis algorithm.

## Environment

To set up the Python environment using `conda` on Ubuntu/macOS:
```setup
git clone git@github.com:xingyuansun/amorsyn.git # clone the code repository
cd amorsyn
conda create -n amorsyn python=3.8
conda activate amorsyn
conda install -c conda-forge fenics # install dependencies
conda install -c conda-forge dolfin-adjoint
conda install pytorch torchvision -c pytorch
pip install matplotlib
pip install tqdm
pip install scipy
pip install ruamel.yaml
pip install tensorboard
pip install meshio
pip install extruder_path/segment_intersection # install the local library
echo "def get_repo_dir():" > repo_dir.py # write repo directory to `repo_dir.py`
echo "    return '$PWD'" >> repo_dir.py
```
Besides, please add the path to this repository to `PYTHONPATH` before executing any Python script:
```setup
export PYTHONPATH=$PYTHONPATH:$PWD
```

## Case study: extruder path planning

### Bullet 3D printing simulator
If you would like to generate your own dataset or evaluate your trained model for extruder path planning,
please download the simulator using the following link and compile it according to its README file: 
[3D printing simulator](https://amorsyn.cs.princeton.edu/bullet-3d-printing.zip).
Make sure to put the path to the 3D printing simulator in the
`get_exec_file_path` function in `extruder_path/yaml_executor.py`.

### Dataset

#### Pre-generated (recommended)
You may directly download the pre-generated datasets using the following links, unzip them,
and put them in the `data` folder:
[carbon fiber dataset](https://amorsyn.cs.princeton.edu/carbon-fiber.zip) (`data/carbon-fiber`),
[Kevlar dataset](https://amorsyn.cs.princeton.edu/kevlar.zip) (`data/kevlar`).
```sh
mkdir data
cd data
wget https://amorsyn.cs.princeton.edu/carbon-fiber.zip
wget https://amorsyn.cs.princeton.edu/kevlar.zip
unzip carbon-fiber.zip
unzip kevlar.zip
rm carbon-fiber.zip
rm kevlar.zip
```

#### Generate your own dataset
Please set up the simulator first (check "Bullet 3D printing simulator").
To generate extruder paths, run 
```sh
python yaml_generator.py
```
in the `extruder_path` folder, with `num_samples` set to the number of paths you need.
The script will automatically generate config files for the simulator for both carbon fiber and Kevlar.
To run the simulator on the generated extruder paths, run
```sh
python yaml_executor.py
```
in the `extruder_path` folder.
You may have to run it multiple times for different materials,
each time specifying a material (`carbon-fiber` or `kevlar`).

### Training and evaluation
Use `amortized_synthesis/train.py` and `amortized_synthesis/test.py`
(make sure you have Bullet 3D printing simulator set up),
which creates a new folder under `amortized_synthesis/runs`, named by timestamp.
The folder contains:
* `opt.json`: config of this run,
* `*.pt` (when running `train.py`): saved PyTorch model, numbered by epoch numbers,
* `results` (when running `test.py`): results on all test samples, with a summary at the end.

In the following, we use carbon fiber as an example, and you may change it to Kevlar.

#### Decoder
Training:
```sh
python train.py --task 3d-printing --model_name decoder --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --pt_material carbon-fiber --num_epochs 10 --learning_rate_decay 0.95
```

#### Encoder
Training:
```sh
python train.py --task 3d-printing --model_name encoder --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --smooth_weight {} --pt_material carbon-fiber --num_epochs 10 --trained_decoder_path {} --learning_rate_decay 0.95
```
with your own `smooth_weight` and `trained_decoder_path`.

Evaluation:
```sh
python test.py --task 3d-printing --model_name encoder --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --pt_material carbon-fiber --trained_model_path {} --pool_size 16
```
with your own `trained_model_path`.

#### `direct-learning`
Training:
```sh
python train.py --task 3d-printing --model_name baseline --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --smooth_weight {} --pt_material carbon-fiber --num_epochs 10 --learning_rate_decay 0.95
```
with your own `smooth_weight`.

Evaluation:
```sh
python test.py --task 3d-printing --model_name baseline --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --pt_material carbon-fiber --trained_model_path {} --pool_size 16
```
with your own `trained_model_path`.

#### `direct-optimization`
Training: we need a trained decoder.

Evaluation:
```sh
python test.py --task 3d-printing --model_name optimizer --mlp_hidden_sizes 500 200 100 50 25 --batch_size 1 --train_val_test_split 9000 500 500 --device 'cpu' --pool_size 16 --smooth_weight {} --pt_material carbon-fiber --trained_model_path {} --bfgs_gtol 1e-7
```
with your own `smooth_weight` and `trained_model_path`.

### Pre-trained models
Pre-trained models of extruder path planning can be downloaded from
[here](https://amorsyn.cs.princeton.edu/path-planning-pretrained.zip).
Note that we provide 3 runs with different random initializations of the neural network.

## Case study: constrained soft robot inverse kinematics

### Dataset

#### Pre-generated (recommended)
You may directly download the pre-generated dataset using the following link, unzip it,
and put it in the `data` folder:
[Soft robot dataset](https://amorsyn.cs.princeton.edu/soft-robot.zip) (`data/soft-robot`).
```sh
mkdir data
cd data
wget https://amorsyn.cs.princeton.edu/soft-robot.zip
unzip soft-robot.zip
rm soft-robot.zip
```

#### Generate your own dataset
If you would like to generate your own dataset, run
```sh
python data_generator.py
```
in the `soft_robot` folder, with `num_dp` set to the number of data samples needed.
You may run it multiple times/simultaneously.

### Training and evaluation
Similarly, use `amortized_synthesis/train.py` and `amortized_synthesis/test.py`.

#### Decoder
Training:
```sh
python train.py --task soft-robot --model_name decoder --mlp_hidden_sizes 128 256 128 --batch_size 8 --train_val_test_split 36000 3000 1000 --device 'cpu' --num_epochs 200 --learning_rate_decay 0.98
```

#### Encoder
Training:
```sh
python train.py --task soft-robot --model_name encoder --mlp_hidden_sizes 128 256 128 --batch_size 8 --train_val_test_split 36000 3000 1000 --device 'cpu' --trained_decoder_path {} --rb_has_obstacle --rb_obstacle_penalty_weight 0.5 --smooth_weight {} --num_epochs 200 --learning_rate_decay 0.98
```
with your own `trained_decoder_path` and `smooth_weight`.

Evaluation:
```sh
python test.py --task soft-robot --model_name encoder --mlp_hidden_sizes 128 256 128 --batch_size 1 --train_val_test_split 36000 3000 1000 --device 'cpu' --trained_model_path {} --rb_has_obstacle
```
with your own `trained_model_path`.

#### `direct-learning`
Training:
```sh
python train.py --task soft-robot --model_name baseline --mlp_hidden_sizes 128 256 128 --batch_size 8 --train_val_test_split 36000 3000 1000 --device 'cpu' --rb_has_obstacle --rb_obstacle_penalty_weight 0.5 --smooth_weight {} --num_epochs 200 --learning_rate_decay 0.98
```
with your own `smooth_weight`.

Evaluation:
```sh
python test.py --task soft-robot --model_name baseline --mlp_hidden_sizes 128 256 128 --batch_size 1 --train_val_test_split 36000 3000 1000 --device 'cpu' --trained_model_path {} --rb_has_obstacle
```
with your own `trained_model_path`.

#### `direct-optimization`
Training: we need a trained decoder.

Evaluation:
```sh
python test.py --task soft-robot --model_name optimizer --mlp_hidden_sizes 128 256 128 --batch_size 1 --train_val_test_split 36000 3000 1000 --device 'cpu' --trained_model_path {} --bfgs_gtol 1e-7 --single_process_optimization --rb_has_obstacle --rb_obstacle_penalty_weight 0.5 --smooth_weight {}
```
with your own `trained_model_path` and `smooth_weight`.

### Pre-trained models
Pre-trained models of constrained soft robot inverse kinematics can be downloaded from
[here](https://amorsyn.cs.princeton.edu/soft-robot-pretrained.zip).
Note that we provide 3 runs with different random initialization of the neural network.

## License
This repository is licensed under the [MIT license](https://github.com/xingyuansun/amorsyn/blob/main/LICENSE).

## README file template and license
README file [template](https://github.com/paperswithcode/releasing-research-code)
and [license](https://github.com/paperswithcode/releasing-research-code/blob/master/LICENSE).
