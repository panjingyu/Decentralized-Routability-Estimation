#!/usr/bin/env python3
"""Split train & validation set on each client device, and preprocess data."""


import argparse
import os
import pickle
import time
from functools import partial, reduce
from multiprocessing.dummy import Pool, Manager
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.contrib import tzip, tmap
from sklearn.model_selection import train_test_split


def preprocess_sample(sample, q, size, device='cpu'):
    """preprocess a sample.

    Clip maximum to the q quantile & resize to size x size.
    """
    if max(sample.shape) > 1500:
        device = 'cpu'
    sample = torch.from_numpy(sample).to(device)
    quant = torch.quantile(sample.flatten(end_dim=1), q=q, dim=0)
    torch.minimum(sample, quant.view(1, 1, -1), out=sample)
    resized_sample = F.interpolate(sample.movedim(-1, 0).unsqueeze(0),
                                   size=size,
                                   mode='bilinear',
                                   align_corners=False)
    return resized_sample.squeeze().movedim(0, -1).cpu().numpy()


def preprocess(features, train_idx, val_idx, size, device='cpu'):
    """Preprocess features.

    1. clip all features to a maximum of .99 quantile of itself
    2. resize to defined size
    3. standardize training & validation sets to [0, 255]
    """
    device = torch.device(device)
    n_train = sum(len(features[idx]) for idx in train_idx)
    n_val = sum(len(features[idx]) for idx in val_idx)

    features_train = chain.from_iterable(features[idx] for idx in train_idx)
    features_val = chain.from_iterable(features[idx] for idx in val_idx)

    sample_preprocessor = partial(preprocess_sample,
                                  q=.99, size=size, device=device)
    features_train = tmap(sample_preprocessor, features_train,
                          desc='preprocess train', total=n_train)
    features_val = tmap(sample_preprocessor, features_val,
                        desc='preprocess val', total=n_val)

    def _get_min_max(tensor):
        tensor = tensor.to(device).flatten(end_dim=1)
        min_, _ = tensor.min(dim=0)
        max_, _ = tensor.max(dim=0)
        return min_, max_

    tensor_train = torch.from_numpy(np.asarray(tuple(features_train)))
    min_train, max_train = zip(*tmap(_get_min_max, tensor_train,
                                     desc='get min & max', total=n_train))
    min_train = reduce(torch.minimum, min_train)
    max_train = reduce(torch.maximum, max_train)
    min_train = min_train.view(1, 1, -1).to(device)
    max_train = max_train.view(1, 1, -1).to(device)

    def _standardize_tensor(tensor):
        tensor = tensor.to(device)
        tensor.add_(-min_train).mul_(255./(max_train-min_train)).round_()
        return tensor.clip_(0., 255.).byte().cpu()

    tensor_train = torch.stack(tuple(tmap(_standardize_tensor, tensor_train,
                                          desc='standardize train',
                                          total=n_train)))

    tensor_val = torch.from_numpy(np.asarray(tuple(features_val)))
    tensor_val = torch.stack(tuple(tmap(_standardize_tensor, tensor_val,
                                        desc='standardize val',
                                        total=n_val)))

    return tensor_train.numpy(), tensor_val.numpy()


def get_label_by_key(labels, indexes, key):
    get_by_key = lambda x: x[key]
    # FIXME: itertools.chain should be iterable, but not working with map
    labels = map(get_by_key,
                 tuple(chain.from_iterable(labels[idx] for idx in indexes)))
    return labels


def get_violated_net_ratio_label(labels, train_idx, val_idx):
    key = 'Incomplete Net Rate'
    labels_train = tuple(get_label_by_key(labels, train_idx, key))
    labels_val = tuple(get_label_by_key(labels, val_idx, key))
    return np.asarray(labels_train), np.asarray(labels_val)


def get_drv_hotspot_label(labels, train_idx, val_idx, size):
    key = 'DRV Density'
    labels_train = get_label_by_key(labels, train_idx, key)
    labels_val = get_label_by_key(labels, val_idx, key)

    def resize_drv_hotspot(label, size=size):
        label = label.astype(np.int32)
        label = torch.from_numpy(label).view(1, 1, *label.shape)
        label = F.interpolate(label.float(), size=size, mode='nearest')
        return label.squeeze().int().clip(0, 1).byte().numpy()
    labels_train = tuple(map(resize_drv_hotspot, labels_train))
    labels_val = tuple(map(resize_drv_hotspot, labels_val))

    return np.asarray(labels_train), np.asarray(labels_val)


def main(args):
    clients = sorted(c for c in os.listdir(args.client_root)
                     if os.path.isdir(os.path.join(args.client_root, c)))
    for client in clients[:-1]:
        manager = Manager()
        client_data = manager.dict()
        client_raw_dir = os.path.join(args.client_root, client, 'raw')
        design_pkls = sorted(s for s in os.listdir(client_raw_dir)
                             if s.endswith('.pkl'))

        def pkl_loader(design_pkl):
            with open(os.path.join(client_raw_dir, design_pkl), 'rb') as f:
                design_name = design_pkl.split('.')[0]
                client_data[design_name] = pickle.load(f)
        start = time.time()
        with Pool(processes=args.n_workers) as pool:
            pool.map(pkl_loader, design_pkls)
        print('{:} loaded, {:.1f} seconds elapsed.'.format(
            client, time.time() - start))

        designs = sorted(client_data.keys())
        train_designs_csv = os.path.join(args.client_root, client,
                                        'train_designs.csv')
        val_designs_csv = os.path.join(args.client_root, client,
                                       'val_designs.csv')
        if not args.split_by_csv:
            train_designs, val_designs = \
                train_test_split(designs, test_size=args.test_size,
                                random_state=args.random_state)
            print('Training & vallidation sets splitted.')

            pd.DataFrame(sorted(train_designs)).to_csv(train_designs_csv,
                                                       index=False,
                                                       header=False)
            pd.DataFrame(sorted(val_designs)).to_csv(val_designs_csv,
                                                     index=False,
                                                     header=False)
            print('Stats dumped.')
            if args.gen_csv_only:
                continue
        else:
            train_designs = pd.read_csv(
                train_designs_csv, header=None
            )[0].tolist()
            val_designs = pd.read_csv(
                val_designs_csv, header=None
            )[0].tolist()

        features = {k: client_data[k][0] for k in client_data}
        labels = {k: client_data[k][1] for k in client_data}
        # reduce to 1/4 if size < 50, reduce to 1/2 if size < 100
        for k in features:
            max_size = max(max(f.shape) for f in features[k])
            print('{}: max size = {:3d}'.format(k, max_size))
            if max_size < 50:
                features[k] = features[k][::4]
                labels[k] = labels[k][::4]
            elif max_size < 100:
                features[k] = features[k][::2]
                labels[k] = labels[k][::2]
        if not args.disable_cuda and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        if client == clients[-1]:
            device = 'cpu'
        print('Preprocess {} on {}.'.format(client, device))
        features_train, features_val = \
            preprocess(features, train_designs, val_designs, args.feature_size,
                       device=device)
        feature_path = os.path.join(args.client_root, client, 'features')
        os.makedirs(feature_path, exist_ok=True)
        np.save(os.path.join(feature_path, 'train.npy'), features_train)
        np.save(os.path.join(feature_path, 'val.npy'), features_val)
        del features_train, features_val
        del features

        labels_vnr_train, labels_vnr_val = \
            get_violated_net_ratio_label(labels, train_designs, val_designs)
        labels_vnr_path = os.path.join(args.client_root, client,
                                       'labels.violated-net-ratio')
        os.makedirs(labels_vnr_path, exist_ok=True)
        np.save(os.path.join(labels_vnr_path, 'train.npy'), labels_vnr_train)
        np.save(os.path.join(labels_vnr_path, 'val.npy'), labels_vnr_val)
        del labels_vnr_train, labels_vnr_val

        labels_drvh_train, labels_drvh_val = \
            get_drv_hotspot_label(labels, train_designs, val_designs,
                                  args.feature_size)
        labels_dvrh_path = os.path.join(args.client_root, client,
                                        'labels.drv-hotspot')
        os.makedirs(labels_dvrh_path, exist_ok=True)
        np.save(os.path.join(labels_dvrh_path, 'train.npy'), labels_drvh_train)
        np.save(os.path.join(labels_dvrh_path, 'val.npy'), labels_drvh_val)
        del labels_drvh_train, labels_drvh_val
        del labels
        del client_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-root', type=str, default='data/alloc',
                        help='Root directory of client data.')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for train/val splitting.')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test size on each client device.')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable GPU acceleration of preprocessing.')
    parser.add_argument('--feature-size', type=int, default=224, metavar='F',
                        help='Feature image size is FxF.')
    parser.add_argument('--n-workers', type=int, default=16,
                        help='Number of CPU workers (for data loading).')
    parser.add_argument('--split-by-csv', action='store_true',
                        help='Split train/val according to existing'
                        'train_designs.csv & val_designs.csv files.')
    parser.add_argument('--gen-csv-only', action='store_true',
                        help='Only generate train_designs.csv '
                             '& val_designs.csv')
    args = parser.parse_args()
    print(args)
    main(args)
