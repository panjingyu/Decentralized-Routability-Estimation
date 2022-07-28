"""I/O tools."""


import os
import pickle
from multiprocessing.dummy import Manager, Process


def load_pkl_to_dict(pkl_dir):
    shard_files = sorted(s for s in os.listdir(pkl_dir) if s.endswith('.pkl'))
    shard_files = [open(os.path.join(pkl_dir, s), 'rb') for s in shard_files]

    manager = Manager()
    data = manager.dict()
    def load_shard(file_obj):
        d_shard = dict()
        while True:
            try:
                d_shard.update(pickle.load(file_obj))
            except EOFError:
                data.update(d_shard)
                del d_shard
                return

    processes = [Process(target=load_shard, args=(f,)) for f in shard_files]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for f in shard_files:
        f.close()

    return data
