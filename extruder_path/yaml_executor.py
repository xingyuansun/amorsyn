import os
import tqdm
from multiprocessing import Pool
from repo_dir import get_repo_dir


def get_exec_file_path():
    return 'path/to/printing_simulator'


def execute_yaml(material):
    output_prefix = os.path.join(get_repo_dir(), 'data', material)
    exec_file_path = get_exec_file_path()
    pool_size = 44
    tasks = []
    with open(os.path.join(output_prefix, 'list')) as log:
        data = log.read()
        data = data.split('\n')
    for item in data:
        if item == '':
            continue
        item = item.split(',')
        assert len(item) == 2
        if os.path.exists(item[1]):
            continue
        cmd_line = '{} {} {} VIS_OFF'.format(exec_file_path, item[0], item[1])
        tasks.append(cmd_line)
    with Pool(pool_size) as p:
        list(tqdm.tqdm(p.imap_unordered(os.system, tasks), total=len(tasks)))


if __name__ == '__main__':
    execute_yaml(material='carbon-fiber')  # carbon-fiber | kevlar
