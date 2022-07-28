#!/usr/bin/env python3
"""Main script."""


import argparse
import copy
import json
import os
import pickle
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.routability import DRVHotspotDataset
from federated.client import Client
from model.pros import PROS
from model.routenet import RouteNetFCN
from model.custom import CompactCNN2
from utils.config import get_server_ckpt
from utils.regularizer import l2_regularizer


def main(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_device_cnt = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        cuda_device_cnt = None

    args.saved = os.path.join('saved', args.saved)
    server_ckpt_path = get_server_ckpt(args.saved, args.finetune_from)
    if not os.path.isfile(server_ckpt_path):
        print('Server checkpoint {} not found, terminate.'.format(args.saved))
        sys.exit(1)
    args.saved = os.path.join(args.saved, 'local_finetuned')
    os.makedirs(args.saved, exist_ok=True)

    n_channels = args.n_channels
    label_size = args.label_size

    use_assigned_cluster = args.cluster_config is not None
    if use_assigned_cluster:
        with open(args.cluster_config, 'r') as f:
            cluster_config = json.load(f)
            print('Cluster assignment:', cluster_config)
        assert args.random_init_cluster is False
    else:
        cluster_config = None

    if args.model == 'routenet224':
        server_model = RouteNetFCN(in_channels=n_channels)
    elif args.model == 'compactcnn2':
        server_model = CompactCNN2(in_channels=n_channels)
    elif args.model == 'pros':
        server_model = PROS(in_channels=n_channels)
    else:
        raise NotImplementedError
    server_load = torch.load(server_ckpt_path)

    # prepare clients
    clients = {}
    val_loaders = []
    device_ids = list(range(cuda_device_cnt))
    for client in os.listdir(args.alloc_dir):
        client_dir = os.path.join(args.alloc_dir, client)
        if not os.path.isdir(client_dir):
            continue
        if cuda_device_cnt is not None:
            client_device = torch.device('cuda:{}'.format(device_ids[0]))
            model = nn.DataParallel(copy.deepcopy(server_model),
                                    device_ids=device_ids)
            model = copy.deepcopy(server_model)
            if cluster_config is not None:
                model.load_state_dict(
                    server_load[cluster_config[client]].state_dict())
            else:
                model.load_state_dict(server_load)
            model = nn.DataParallel(model,
                                    device_ids=device_ids)
            device_ids = device_ids[1:] + device_ids[:1]
        data_train = DRVHotspotDataset(client_dir,
                                       train=True,
                                       n_channels=n_channels,
                                       label_size=label_size)
        data_val = DRVHotspotDataset(client_dir,
                                     train=False,
                                     n_channels=n_channels,
                                     label_size=label_size)
        loader_train = DataLoader(data_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_workers,
                                  pin_memory=True)
        loader_val = DataLoader(data_val,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                pin_memory=True)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        client_config = {
            'model': model,
            'server': None,
            'loader_train': loader_train,
            'loader_val': loader_val,
            'optimizer': optimizer,
            'regularizer': l2_regularizer,
            'reg_strength': args.l2_strength,
            'criterion': nn.BCEWithLogitsLoss(),
            'device': client_device,
        }
        clients[client] = Client(**client_config)
        val_loaders.append(loader_val)

    # finetune
    loss, acc, auc = {}, {}, {}
    for client_idx in clients:
        loss[client_idx] = []
        acc[client_idx] = []
        auc[client_idx] = []
    start = time.time()
    for round_idx in range(args.max_round):
        selected_clients = sorted(clients.keys())
        for client_idx in selected_clients:
            client = clients[client_idx]
            client.model.to(client.device)
            print('[Round {}/{}] Training {}'.format(
                round_idx+1, args.max_round, client_idx
            ), end=' -> ')
            client.train_fedprox_one_round(args.round_steps,
                                           fedprox_mu=0.,
                                           report_freq=None)
            client.model.to('cpu')
            torch.save(client.model,
                       os.path.join(args.saved, '{}-{}.pt'.format(
                           client_idx, round_idx+1
                       )))
        if (round_idx + 1) % args.val_freq == 0:
            print('- - - -')
            round_auc = []
            for client_idx in selected_clients:
                client = clients[client_idx]
                print('[Round {}/{}] Val {}'.format(
                    round_idx+1, args.max_round, client_idx
                ), end=' -> ')
                client.model.to(client.device)
                _loss, _acc, _auc = client.validate_one_epoch(report_freq=None)
                loss[client_idx].append(_loss)
                acc[client_idx].append(_acc)
                auc[client_idx].append(_auc)
                pickle.dump(loss,
                            open(os.path.join(args.saved, 'loss.pkl'), 'wb'))
                pickle.dump(acc,
                            open(os.path.join(args.saved, 'acc.pkl'), 'wb'))
                pickle.dump(auc,
                            open(os.path.join(args.saved, 'auc.pkl'), 'wb'))
                round_auc.append(_auc)
            print('Validation done, avg auc = {:.3f}'.format(
                np.mean(round_auc)))

        end = time.time()
        print('Round elapsed time: {:.2f} mins\n'.format((end - start)/60))
        start = end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable GPU and use only CPU.')
    parser.add_argument('--n-workers', type=int, default=16,
                        help='Number of CPU workers.')
    parser.add_argument('--alloc-dir', type=str, default='data/alloc.src-wise',
                        help='Directory of allocated data.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for numpy RNG.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training & validation.')
    parser.add_argument('--l2-strength', type=float, default=1e-3,
                        help='L2 regularization strength.')
    parser.add_argument('--saved', type=str, default='fedavg',
                        help='Directory of saved results.')
    parser.add_argument('--model', type=str, default='routenet')
    parser.add_argument('--n-channels', type=int, default=64,
                        help='Number of input channels.')
    parser.add_argument('--label-size', type=int, default=224,
                        help='Length of label maps.')
    parser.add_argument('--max-round', type=int, default=100,
                        help='Max finetuning epochs.')
    parser.add_argument('--round-steps', type=int, default=100,
                        help='#steps of training per round.')
    parser.add_argument('--random-init-cluster', action='store_true',
                        help='Clients adpot random clusters at the 1st round')
    parser.add_argument('--cluster-config', type=str, default=None,
                        help='Config json file assigning cluster to clients explicitly')
    parser.add_argument('--val-freq', type=int, default=5,
                        help='Validation frequency.')
    parser.add_argument('--finetune-from', type=int, default=None,
                        help='Finetune from which round.')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    main(args)
