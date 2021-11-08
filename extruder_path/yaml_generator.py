import os
import math
import tqdm
import ruamel.yaml
import numpy as np
from multiprocessing import Pool
from extruder_path.random_path_generator import random_jordan_curve
from repo_dir import get_repo_dir


def chirp_path(config, A, a, b, c, xFrom, xTo, pathDiscreRes):
    config['printerPathX'], config['printerPathY'], config['printerPathZ'] = [], [], []
    for i in range(pathDiscreRes):
        x = xFrom + (xTo - xFrom) / (pathDiscreRes - 1.) * i
        y = A * math.sin(a * x * x + b * x + c)
        config['printerPathX'].append(x)
        config['printerPathZ'].append(y)


def update_dependent_config(config, eps=1e-4):
    config['groundStickyHeightThreshold'] = config['ropeBodyHeight'] / 2. + eps
    config['simulationStopHeightThreshold'] = config['ropeBodyHeight'] / 2. + eps
    config['simulationInitHeightThreshold'] = config['printerOriginY'] - config['printerHeight'] / 2.
    height = config['printerOriginY'] + config['printerHeight'] / 2. + config['printerCollisionMargin'] + \
             config['ropeBodyHeight'] / 2.
    config['ropePathY'] = [0., height, height]
    config['printerPathY'] = [config['printerOriginY'] for _ in range(len(config['printerPathX']))]


def config_init():
    config = {}

    # set default config
    # rope options
    config['ropeResolution'] = 20
    config['ropeIndividualMass'] = 1.
    config['ropeBodyHeight'] = 0.005
    config['ropeP2PERP'] = .8
    config['ropeEnableSpring'] = True
    config['ropeSpringStiffness'] = 20.
    config['ropeSpringDamping'] = 1.
    config['ropeSpringERP'] = .8

    # ground options
    config['groundFrictionCoefficient'] = 100.
    config['groundEnableSticky'] = False
    config['groundStickyHeightThreshold'] = -1.
    config['groundStickyStiffness'] = .01
    config['groundStickyDamping'] = 0.
    config['groundStickyERP'] = .6

    # printer options
    config['printerHeight'] = 0.0275
    config['printerCollisionMargin'] = 0.025
    config['printerMass'] = 1000.
    config['printerOriginX'] = 0.
    config['printerOriginY'] = .055
    config['printerOriginZ'] = 0.
    config['printerFrictionCoefficient'] = 1.45

    # simulation options
    config['simulationTimeStepSize'] = .005
    config['simulationStopHeightThreshold'] = -1.
    config['simulationInitHeightThreshold'] = -1.
    config['simulationPrintingSpeed'] = .06
    config['simulationLengthLimit'] = 4.35
    # config['simulationLengthLimit'] = 4.65
    config['simulationTimeLimit'] = 200.

    # rope path
    config['ropePathX'] = [0., 0., 0.15]
    config['ropePathY'] = []
    config['ropePathZ'] = [0., 0., 0.]

    # printer path
    config['printerPathX'], config['printerPathY'], config['printerPathZ'] = [], [], []
    return config


def path_save_to(path, yml_path, stop_sim_factor=None, override=None):
    config = config_init()
    if override is not None:
        for key, value in override.items():
            config[key] = value
    yaml = ruamel.yaml.YAML()
    assert np.abs(path[0][0]) + np.abs(path[0][1]) < 1e-5
    for x, y in path:
        config['printerPathX'].append(x.item())
        config['printerPathZ'].append(y.item())
    update_dependent_config(config)
    if stop_sim_factor is not None:
        config['simulationStopHeightThreshold'] = config['ropeBodyHeight'] * stop_sim_factor
    with open(yml_path, 'w') as f:
        yaml.dump(config, f)


def get_material_para(material):
    if material == 'carbon-fiber':
        return {'ropeSpringStiffness': 15.2, 'printerFrictionCoefficient': 1.81}
    elif material == 'kevlar':
        return {'ropeSpringStiffness': 15, 'printerFrictionCoefficient': 1.51}
    else:
        raise Exception()


def random_path_save_to(yml_paths):
    c_yml_path, k_yml_path = yml_paths
    random_curve = random_jordan_curve()
    random_curve -= random_curve[0:1]
    path_save_to(random_curve, c_yml_path, override=get_material_para('carbon-fiber'))
    path_save_to(random_curve, k_yml_path, override=get_material_para('kevlar'))


def save_list_file(output_dir, num_samples):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    yml_paths = []
    with open(os.path.join(output_dir, 'list'), 'w') as log:
        for i in range(num_samples):
            yml_path = os.path.join(output_dir, '{:05d}.yml'.format(i))
            rst_path = os.path.join(output_dir, '{:05d}.txt'.format(i))
            log.write(yml_path + ',' + rst_path + '\n')
            yml_paths.append(yml_path)
    return yml_paths


def generate_yaml(num_samples, pool_size):
    data_dir = os.path.join(get_repo_dir(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    c_output_dir = os.path.join(data_dir, 'carbon-fiber')
    k_output_dir = os.path.join(data_dir, 'kevlar')
    c_yml_paths = save_list_file(c_output_dir, num_samples=num_samples)
    k_yml_paths = save_list_file(k_output_dir, num_samples=num_samples)
    args = list(zip(c_yml_paths, k_yml_paths))
    with Pool(pool_size) as p:
        list(tqdm.tqdm(p.imap_unordered(random_path_save_to, args), total=len(args)))


if __name__ == '__main__':
    generate_yaml(num_samples=10000, pool_size=44)
