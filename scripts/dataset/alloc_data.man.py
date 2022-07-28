#!/usr/bin/env python3
"""Script for allocating raw data to multiple client devices."""


import argparse
import os

import numpy as np
import pandas as pd


def main(args):
    clients = sorted(os.path.join(args.alloc_dir, d)
                     for d in os.listdir(args.alloc_dir)
                     if os.path.isdir(os.path.join(args.alloc_dir, d)))
    for c in clients:
        client_designs_csv = os.path.join(c, 'designs.csv')
        if os.path.isfile(client_designs_csv):
            designs = pd.read_csv(client_designs_csv, header=None)[0].tolist()
            if all(os.path.isfile(os.path.join(args.raw_dir, d+'.pkl'))
                   for d in designs):
                client_raw_dir = os.path.join(c, 'raw')
                os.makedirs(client_raw_dir, exist_ok=True)
                # unsure empty folder
                for f in os.listdir(client_raw_dir):
                    os.remove(os.path.join(client_raw_dir, f))
                # link raw pkl files to client raw folder
                for d in designs:
                    os.link(os.path.join(args.raw_dir, d+'.pkl'),
                            os.path.join(client_raw_dir, d+'.pkl'))
            else:
                print('Error in {}'.format(c))
                print('some designs not found in', args.raw_dir)
                exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=str,
                        default='./data/raw.src-wise',
                        help='Directory of raw features & labels '
                             'stored in pickle files.')
    parser.add_argument('--alloc-dir', type=str, default='./data/alloc',
                        help='Directory of allocated data.')
    parser.add_argument('--alloc-ext', '-e', type=str,
                        default='src-wise.man-v1',
                        help='Extension to directory of allocated data.')
    args = parser.parse_args()
    if args.alloc_ext:
        args.alloc_dir = '.'.join((args.alloc_dir, args.alloc_ext))
    print(args)

    np.random.seed(42)
    main(args)
