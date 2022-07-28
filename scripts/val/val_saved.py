#!/usr/bin/env python3
"""Validate saved model."""

import argparse
import collections
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.routability import DRVHotspotDataset
from model.custom import CompactCNN2
from model.pros import PROS
from model.routenet import RouteNetFCN
from utils.metrics import roc_auc


def get_state_dict(ckpt, mode='global'):
    state_dict = torch.load(ckpt, map_location='cpu')
    if not isinstance(state_dict, collections.OrderedDict):
        print(type(state_dict))
        print('Double check loaded type')
        exit()
    # new_state_dict
    # for key in state_dict:
    #     if key.startswith('module.'):
    #         no_module_key = key[7:]
    #         state_dict[no_module_key] = state_dict.pop(key)
    return state_dict

def roc_auc_per_sample(actual, predicted, label_size=224, device='cpu'):
    assert actual.shape == predicted.shape
    n_samples = actual.shape[0]
    auc = 0.
    for i in range(n_samples):
        if actual[i].max() > 0:
            auc += roc_auc(actual[i].ravel(), predicted[i].ravel(),
                        device=device)
    auc /= n_samples
    return auc


def main(args):
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    saved = os.path.join('saved', args.saved)
    assert os.path.isdir(saved), 'Saved model not found in {}!'.format(saved)
    assert args.resume_from is not None, \
        'Specify which model to run validation!'
    if args.mode == 'global':
        ckpt = os.path.join(saved, 'global-{}.pt')
    elif args.mode == 'global-archive':
        ckpt = os.path.join(saved, 'server-{}.pt')
    else:
        raise NotImplementedError('Specify the correct mode.')
    ckpt = ckpt.format(args.resume_from)
    assert os.path.isfile(ckpt), 'No such checkpoint: {}'.format(ckpt)
    print('Found checkpoint:', ckpt)

    label_size = args.label_size
    n_channels = args.n_channels
    batch_size = 32

    if args.model.lower() == 'routenet224':
        server = RouteNetFCN(in_channels=n_channels)
    elif args.model.lower() == 'pros':
        server = PROS(in_channels=n_channels)
    elif args.model.lower() == 'compactcnn2' \
         or args.model.lower() == 'compact2':
        server = CompactCNN2(in_channels=n_channels)
    else:
        raise NotImplementedError
        
    cuda_device_cnt = torch.cuda.device_count()

    # prepare clients
    clients = {}
    device_ids = list(range(cuda_device_cnt))
    for client in sorted(os.listdir(args.alloc_dir)):
        client_dir = os.path.join(args.alloc_dir, client)
        if not os.path.isdir(client_dir):
            continue
        model = copy.deepcopy(server)
        if cuda_device_cnt > 1 and use_cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            client_device = torch.device('cuda:{}'.format(device_ids[0]))
            device_ids = device_ids[1:] + device_ids[:1]
        else:
            model = nn.DataParallel(model)
            client_device = device
        state_dict = get_state_dict(ckpt, args.mode)
        model.load_state_dict(state_dict)

        data_val = DRVHotspotDataset(client_dir,
                                     train=False,
                                     label_size=label_size,
                                     n_channels=n_channels)
        loader_val = DataLoader(data_val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                pin_memory=False)
        
        model.eval()
        total_samples = 0
        total = correct = 0
        pred_epoch, target_epoch = [], []
        with torch.no_grad():
            for data, target in loader_val:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # prediction = output.greater(0.)
                pred_epoch.append(output.cpu().numpy())
                target_epoch.append(target.cpu().numpy())
                # total += target.numel()
                # correct += prediction.eq(target).sum()
                total_samples += target.shape[0]
            pred_epoch = np.vstack(pred_epoch)
            target_epoch = np.vstack(target_epoch)
            auc_score = roc_auc_per_sample(target_epoch,
                                           pred_epoch,
                                           label_size=label_size,
                                           device=device)
            auc_score_total = roc_auc(target_epoch.ravel(),
                                      pred_epoch.ravel(),
                                      device=device)
        print(client, '-> auc = {:.4f}'.format(auc_score))
        print(client, '-> auc = {:.4f} (total)'.format(auc_score_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable GPU and use only CPU.')
    parser.add_argument('--n-workers', type=int, default=16,
                        help='Number of CPU workers.')
    parser.add_argument('--alloc-dir', type=str, default='data/alloc.src-wise',
                        help='Directory of allocated data.')
    parser.add_argument('--test-freq', type=int, default=5, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='ifca',
                        help='Directory of saved results.')
    parser.add_argument('--model', type=str, default='routenet')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--label-size', type=int, default=56,
                        help='Length of label map.')
    parser.add_argument('--n-channels', type=int, default=64,
                        help='Number of input channels.')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of clusters')
    parser.add_argument('--id-init', action='store_true',
                        help='Use identical init on cluster models')
    parser.add_argument('--random-init-cluster', action='store_true',
                        help='Clients adpot random clusters at the 1st round')
    parser.add_argument('--cluster-config', type=str, default=None,
                        help='Config json file assigning cluster to clients explicitly')
    parser.add_argument('--resume-from', type=int, default=None,
                        help='The round to validate.')
    args = parser.parse_args()
    print(args)
    main(args)