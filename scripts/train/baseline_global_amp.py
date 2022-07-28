#!/usr/bin/env python3
"""Main script."""


import argparse
import copy
import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from dataset.routability import DRVHotspotDataset
from federated.client import Client
from federated.server import fed_avg
from model.routenet import RouteNetFCN
from model.pros import PROS
from utils.regularizer import l2_regularizer


def main(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.saved = os.path.join('saved', args.saved)
    os.makedirs(args.saved, exist_ok=True)

    # server = nn.DataParallel(RouteNetFCN(in_channels=64).to(device))
    server = nn.DataParallel(PROS(in_channels=64).to(device))

    # prepare clients
    clients = {}
    client_dirs = (os.path.join(args.alloc_dir, client) 
                   for client in os.listdir(args.alloc_dir))
    client_dirs = sorted(d for d in client_dirs if os.path.isdir(d))
    model = copy.deepcopy(server)
    model = model
    data_train = DRVHotspotDataset(client_dirs, train=True)
    datas_val = [DRVHotspotDataset(d, train=False) for d in client_dirs]
    loader_train = DataLoader(data_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers,
                              pin_memory=True)
    loaders_val = [DataLoader(d,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.n_workers,
                              pin_memory=True)
                   for d in datas_val]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)
    client_config = {
        'model': model,
        'server': server,
        'loader_train': loader_train,
        'loader_val': None,
        'optimizer': optimizer,
        'regularizer': l2_regularizer,
        'reg_strength': args.l2_strength,
        'criterion': nn.BCEWithLogitsLoss(),
        'device': device,
        'grad_scaler': amp.GradScaler(),
    }
    clients['global'] = Client(**client_config)


    # train & validation
    loss, acc, auc = {}, {}, {}
    for round_idx in range(args.max_round):
        print('\nRound {}/{}'.format(round_idx + 1, args.max_round))
        selected_clients = sorted(clients.keys())
        for client_idx in selected_clients:
            client = clients[client_idx]
            # client.fetch_server()
            print('\nTraining', client_idx)
            client.train_one_epoch(args.round_steps, with_amp=True)
            print('\nValidation')
            for idx, loader in enumerate(loaders_val):
                loss_, acc_, auc_ = client.validate_one_epoch(loader=loader)
                if idx not in loss:
                    loss[idx] = [loss_]
                    acc[idx] = [acc_]
                    auc[idx] = [auc_]
                else:
                    loss[idx].append(loss_)
                    acc[idx].append(acc_)
                    auc[idx].append(auc_)

        pickle.dump(loss, open(os.path.join(args.saved, 'loss.pkl'), 'wb'))
        pickle.dump(acc, open(os.path.join(args.saved, 'acc.pkl'), 'wb'))
        pickle.dump(auc, open(os.path.join(args.saved, 'auc.pkl'), 'wb'))

        # local training done, perform FedAvg
        client_states = [c.model.state_dict() for c in clients.values()]
        print('\nFed Avg')
        fed_avg(client_states, server)
        torch.save(
            server.state_dict(),
            os.path.join(args.saved, 'server-{}.pt'.format(round_idx+1))
        )


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
    parser.add_argument('--max-round', type=int, default=20,
                        help='Maximum number of rounds.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training & validation.')
    parser.add_argument('--l2-strength', type=float, default=1e-2,
                        help='L2 regularization strength.')
    parser.add_argument('--round-steps', type=int, default=100,
                        help='#steps of training per round.')
    parser.add_argument('--test-freq', type=int, default=2, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='baseline.global',
                        help='Directory of saved results.')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    main(args)
