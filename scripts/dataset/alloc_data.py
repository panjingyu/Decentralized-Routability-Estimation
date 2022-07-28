#!/usr/bin/env python3
"""Script for allocating data to multiple client devices."""


import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import load_blacklist, load_yaml
from utils.io import load_pkl_to_dict


def arrange_data(data):
    # load blacklists
    layout_blacklist = load_blacklist('config/layout_blacklist.csv')
    design_blacklist = load_blacklist('config/design_blacklist.csv')
    print('Layout blacklist:', layout_blacklist)
    print('Design blacklist:', design_blacklist)

    features, labels = {}, {}
    layout_opts = {}
    design_cnt = 0
    for k, (ft, lb) in data.items():
        if any(k.startswith(l) for l in layout_blacklist):
            continue
        if any(k.endswith(d) for d in design_blacklist):
            continue
        opt_name, design_name = k.split('/')
        ft = np.asarray(ft)
        if design_name not in features:
            features[design_name] = [ft]
            labels[design_name] = [lb]
            layout_opts[design_name] = [opt_name]
            design_cnt += 1
        else:
            features[design_name].append(ft)
            labels[design_name].append(lb)
            layout_opts[design_name].append(opt_name)
    return features, labels, layout_opts


def dump_design_stats(features, designs, benchmarks, out='stats.csv'):
    stats = {'design': sorted(designs)}
    stats['benchmark'] = []
    stats['#samples'] = []
    stats['max size'] = []
    stats['min size'] = []
    for design in stats['design']:
        stats['#samples'].append(len(features[design]))
        stats['max size'].append(max(f.shape[0] for f in features[design]))
        stats['min size'].append(min(f.shape[0] for f in features[design]))
        benchmark_match = None
        for bm in benchmarks:
            if design in benchmarks[bm]:
                benchmark_match = bm
                break
        stats['benchmark'].append(benchmark_match)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(stats).to_csv(out, index=False)


def main(args):
    # load pkl data
    start = time.time()
    data = load_pkl_to_dict(args.raw_dir)
    t_elapsed = time.time() - start
    print('{0} samples loaded, {1:.2f}s elapsed.'.format(len(data), t_elapsed))

    features, labels, layout_opts = arrange_data(data)
    del data

    # load benchmark specs
    config = load_yaml(args.benchmarks_yaml)

    designs = sorted(features.keys())
    if not config.clients:
        # random allocation
        np.random.shuffle(designs)
        n_designs_avg = len(designs) // args.n_clients
        designs_of_client = []
        alloc_design_cnt = 0
        for i in range(args.n_clients):
            if i < args.n_clients - 1:
                n_designs = n_designs_avg
            else:
                n_designs = len(designs) - (args.n_clients - 1) * n_designs_avg
            designs_of_client.append(
                designs[alloc_design_cnt:alloc_design_cnt+n_designs])
            alloc_design_cnt += n_designs
        print('#designs per client:', [len(d) for d in designs_of_client])
    else:
        args.n_clients = len(config.clients)
        designs_of_client = []
        for client_sources in config.clients:
            designs_match = [
                d for d in designs
                if any(d in config.benchmark[src] for src in client_sources)
            ]
            designs_of_client.append(designs_match)

    dump_design_stats(features, designs, config.benchmark,
                      out=os.path.join(args.alloc_dir, 'designs.csv'))

    for i in range(args.n_clients):
        client_name = 'client_{}'.format(i)
        client_dir = os.path.join(args.alloc_dir, )
        client_raw_dir = os.path.join(client_dir, 'raw')
        client_designs = designs_of_client[i]

        designs_log = os.path.join(client_dir, 'designs.csv')
        dump_design_stats(features, client_designs, config.benchmark,
                          out=designs_log)

        os.makedirs(client_raw_dir, exist_ok=True)
        for c_design in tqdm(client_designs, desc='alloc '+client_name):
            design_pkl_path = os.path.join(client_raw_dir,
                                           '{}.pkl'.format(c_design))
            with open(design_pkl_path, 'wb') as f:
                pickle.dump((features[c_design],
                             labels[c_design],
                             layout_opts[c_design]),
                            f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clients', type=int, default=3,
                        help='Number of client devices.')
    parser.add_argument('--raw-dir', type=str,
                        default='./data/raw-64.0423/_feature-pkl',
                        help='Directory of raw features & labels '
                             'stored in pickle files.')
    parser.add_argument('--alloc-dir', type=str, default='./data/alloc',
                        help='Directory of allocated data.')
    parser.add_argument('--alloc-ext', '-e', type=str, default='',
                        help='Extension to directory of allocated data.')
    parser.add_argument('--benchmarks-yaml', type=str,
                        default='config/benchmarks.yml',
                        help='Yaml file that specifies benchmarks.')
    args = parser.parse_args()
    if args.alloc_ext:
        args.alloc_dir = '.'.join((args.alloc_dir, args.alloc_ext))
    print(args)

    np.random.seed(42)
    main(args)
