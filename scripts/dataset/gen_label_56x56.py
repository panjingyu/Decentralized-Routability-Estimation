#!/usr/bin/env python3
"""Generate 56x56 labels from original 224x224 ones."""


import argparse
import os
import sys

import numpy as np


def convert_label_224_to_56(labels_224):
    assert len(labels_224.shape) == 3
    assert labels_224.shape[1] == labels_224.shape[2]
    assert labels_224.shape[1] == 224
    labels_56 = np.empty((labels_224.shape[0], 56, 56),
                         dtype=np.uint8)
    for i in range(56):
        for j in range(56):
            labels_56[:,i,j] = labels_224[:,4*i:4*i+4, 4*j:4*j+4]\
                               .max(axis=(1,2))
    return labels_56


def main(args):
    clients = sorted(c for c in os.listdir(args.client_root)
                     if os.path.isdir(os.path.join(args.client_root, c)))
    print(clients)
    for client in clients:
        label_224_dir = os.path.join(args.client_root,
                                     client,
                                     'labels.drv-hotspot')
        if not os.path.isdir(label_224_dir):
            print(label_224_dir, 'not found!')
            sys.exit()
        label_56_dir = os.path.join(args.client_root,
                                    client,
                                    'labels.drv-hotspot.56x56')
        os.makedirs(label_56_dir, exist_ok=True)
        label_train = np.load(os.path.join(label_224_dir, 'train.npy'))
        label_val = np.load(os.path.join(label_224_dir, 'val.npy'))
        # nearest
        label_56_train = convert_label_224_to_56(label_train)
        label_56_val = convert_label_224_to_56(label_val)
        np.save(os.path.join(label_56_dir, 'train.npy'), label_56_train)
        np.save(os.path.join(label_56_dir, 'val.npy'), label_56_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-root', type=str, default='data/alloc',
                        help='Root directory of client data.')
    parser.add_argument('--feature-size', type=int, default=224, metavar='F',
                        help='Feature image size is FxF.')
    parser.add_argument('--n-workers', type=int, default=16,
                        help='Number of CPU workers (for data loading).')
    args = parser.parse_args()
    main(args)
