import os
import shutil
import tempfile
import time
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from repo_dir import get_repo_dir
from soft_robot.arguments import get_args
from soft_robot.trainer_robot import TrainerRobot


def soft_robot_limit():
    return 0.2


def soft_robot_worker(argument):
    if isinstance(argument[0][1], str):
        tmp_dir = None
        sources = [item[0] for item in argument]
        saving_paths = [item[1] for item in argument]
    else:
        tmp_dir = tempfile.mkdtemp()
        sources = argument
        saving_paths = [os.path.join(tmp_dir, 'u')] * len(sources)

    args = get_args()
    trainer = TrainerRobot(args)
    displacements = trainer.simulate(sources, saving_paths)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)
    return displacements


def soft_robot_dispatcher(sources, pool_size, saving_paths=None):
    if saving_paths is None:
        data = sources
    else:
        assert len(sources) == len(saving_paths)
        data = list(zip(sources, saving_paths))
    if pool_size == 1:
        return soft_robot_worker(data)
    split_sources = np.array_split(data, pool_size)
    with Pool(pool_size) as p:
        displacements = p.map(soft_robot_worker, split_sources)
    return np.concatenate(displacements)


def soft_robot_data_generation(pool_size, num_dp, a):
    data_dir = os.path.join(get_repo_dir(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    saving_dir = os.path.join(data_dir, 'soft-robot')
    sources = np.random.uniform(-a, a, size=(num_dp, 40))
    displacements = soft_robot_dispatcher(sources=sources, pool_size=pool_size, saving_paths=None)
    time_stamp = time.time()
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    np.save(os.path.join(saving_dir, 'input_{}.npy'.format(time_stamp)), sources)
    np.save(os.path.join(saving_dir, 'output_{}.npy'.format(time_stamp)), displacements)


def soft_robot_data_checker(data_dir, time_stamp, eps=1e-1):
    sources = np.load(os.path.join(data_dir, 'input_{}.npy'.format(time_stamp)))
    displacements = np.load(os.path.join(data_dir, 'output_{}.npy'.format(time_stamp)))
    assert len(sources) == len(displacements)
    tmp_dir = tempfile.mkdtemp()
    for idx in tqdm(range(len(sources))):
        source = sources[idx]
        args = get_args()
        trainer = TrainerRobot(args)
        displacement = trainer.simulate(sources=[source], saving_paths=[os.path.join(tmp_dir, 'u')])[0]
        mean_dis = np.mean(np.sum((displacements[idx] - displacement) ** 2, axis=1) ** 0.5)
        print(mean_dis)
        assert mean_dis < eps
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    soft_robot_data_generation(pool_size=1, num_dp=10000, a=soft_robot_limit())
