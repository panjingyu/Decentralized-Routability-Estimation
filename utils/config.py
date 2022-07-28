"""Modules for configuration setting."""


import os
import yaml


class Config:
    def __init__(self, d):
        self.__dict__ = d


def load_blacklist(csv_path):
    with open(csv_path, 'r') as f:
        blacklist = [line.split('#')[0].strip() for line in f]
    return blacklist


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return Config(config)

def get_server_ckpt(saved_dir, ckpt_num=None):
    checkpoints = os.listdir(saved_dir)
    if ckpt_num is None:
        round_nums = sorted(
            int(ckpt.split('.')[0].split('-')[1])
            for ckpt in checkpoints if ckpt.endswith('.pt'))
        ckpt_num = max(round_nums)
    ckpt = 'server-{}.pt'.format(ckpt_num)
    if ckpt not in checkpoints:
        ckpt = 'global-{}.pt'.format(ckpt_num)
    assert ckpt in checkpoints, 'Checkpoint {} not found in {}!'.format(
        ckpt, saved_dir)
    latest_path = os.path.join(saved_dir, ckpt)
    print('Get latest server: {}'.format(latest_path))
    return latest_path

def get_latest_finetuning(saved_dir, n_clients):
    finetuning_dir = os.path.join(saved_dir, 'local_finetuned')
    checkpoints = os.listdir(finetuning_dir)
    latest_paths = []
    for i in range(1, n_clients + 1):
        max_finetuning_round = max(
            int(ckpt.split('.')[0].split('-')[-1])
            for ckpt in checkpoints
            if ckpt.startswith('client-{}'.format(i)) and ckpt.endswith('.pt')
        )
        latest_paths.append(os.path.join(finetuning_dir,
                                         'client-{}-{}.pt'.format(
                                             i, max_finetuning_round
                                         )))
    return latest_paths


if __name__ == '__main__':
    with open('config/benchmarks.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    print(config)
