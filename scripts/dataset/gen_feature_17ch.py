#!/usr/bin/env python3
"""Generate feautres from 64 channels to 17 channels."""

import os
import sys

import numpy as np


def main():
    client_root = 'data/alloc'
    clients = sorted(c for c in os.listdir(client_root)
                     if os.path.isdir(os.path.join(client_root, c)))
    print(clients)
    for client in clients:
        print('Processing', client)
        feature_64ch_dir = os.path.join(client_root,
                                        client,
                                        'features')
        feature_17ch_dir = os.path.join(client_root,
                                        client,
                                        'features.17ch')
        if not os.path.isdir(feature_64ch_dir):
            print(feature_64ch_dir, 'not found!')
            sys.exit()
        os.makedirs(feature_17ch_dir, exist_ok=True)
        feature_train = np.load(os.path.join(feature_64ch_dir, 'train.npy'))
        feature_val = np.load(os.path.join(feature_64ch_dir, 'val.npy'))
        # what channels we use in the NAS paper
        ch_17 = [
            0, 1, 2, 6, 18, 19, 23, 24, 43, 44, 48, 49, 53, 54, 58, 59, -1
        ]
        assert len(ch_17) == 17
        feature_17_train = np.stack([feature_train[..., c] for c in ch_17],
                                    axis=3)
        feature_17_val = np.stack([feature_val[..., c] for c in ch_17],
                                  axis=3)
        np.save(os.path.join(feature_17ch_dir, 'train.npy'), feature_17_train)
        np.save(os.path.join(feature_17ch_dir, 'val.npy'), feature_17_val)


if __name__ == '__main__':
    main()
