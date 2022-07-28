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

from dataset.routability import DRVHotspotDataset
from federated.client import Client
from model.pros import PROS
from model.routenet import RouteNetFCN
from utils.regularizer import l2_regularizer


def main(args):
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.saved = os.path.join('saved', args.saved)
    os.makedirs(args.saved, exist_ok=True)

    if args.model.lower() == 'routenet':
        server = RouteNetFCN(in_channels=64)
    elif args.model.lower() == 'pros':
        server = PROS(in_channels=64)
        
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
            client_device = device
        if args.resume_from is not None:
            client_ckpt = os.path.join(
            print(client, 'loaded from', client_ckpt)
            model.load_state_dict(torch.load(client_ckpt))

        data_train = DRVHotspotDataset(client_dir, train=True)
        data_val = DRVHotspotDataset(client_dir, train=False)
        loader_train = DataLoader(data_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_workers,
                                  pin_memory=False)
        loader_val = DataLoader(data_val,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                pin_memory=False)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        client_config = {
            'model': model,
            'server': server,
            'loader_train': loader_train,
            'loader_val': loader_val,
            'optimizer': optimizer,
            'regularizer': l2_regularizer,
            'reg_strength': args.l2_strength,
            'criterion': nn.BCEWithLogitsLoss(),
            'device': client_device,
        }
        clients[client] = Client(**client_config)

    # train & validation
    if args.resume_from is not None:
        loss = pickle.load(open(os.path.join(args.saved, 'loss.pkl'), 'rb'))
        acc = pickle.load(open(os.path.join(args.saved, 'acc.pkl'), 'rb'))
        auc = pickle.load(open(os.path.join(args.saved, 'auc.pkl'), 'rb'))
    else:
        loss, acc, auc = {}, {}, {}
    start_round = args.resume_from if args.resume_from is not None else 0
    for round_idx in range(start_round, args.max_round):
        print('\nRound {}/{}'.format(round_idx + 1, args.max_round))
        selected_clients = sorted(clients.keys())
        for client_idx in selected_clients:
            client = clients[client_idx]
            client.model.to(client.device)
            print('\nTraining', client_idx)
            client.train_one_epoch(args.round_steps)
            for val_idx in sorted(clients.keys()):
                if round_idx + 1 < args.max_round:
                    continue
                # if val_idx != client_idx:
                #     continue
                print('\nValidate', client_idx, 'on', val_idx)
                loss_, acc_, auc_ = client.validate_one_epoch(
                    loader=clients[val_idx].loader_val)
                if (client_idx, val_idx) not in loss:
                    loss[(client_idx, val_idx)] = [loss_]
                    acc[(client_idx, val_idx)] = [acc_]
                    auc[(client_idx, val_idx)] = [auc_]
                else:
                    loss[(client_idx, val_idx)].append(loss_)
                    acc[(client_idx, val_idx)].append(acc_)
                    auc[(client_idx, val_idx)].append(auc_)
            client_saved_dir = os.path.join(args.saved, client_idx)
            os.makedirs(client_saved_dir, exist_ok=True)
            client.model.to('cpu')
            # if round_idx + 1 == args.max_round:
            # torch.save(client.model.state_dict(),
            #            os.path.join(client_saved_dir,
            #                         'round-{}.pt'.format(round_idx + 1)))


        pickle.dump(loss, open(os.path.join(args.saved, 'loss.pkl'), 'wb'))
        pickle.dump(acc, open(os.path.join(args.saved, 'acc.pkl'), 'wb'))
        pickle.dump(auc, open(os.path.join(args.saved, 'auc.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable GPU and use only CPU.')
    parser.add_argument('--model', type=str, default='routenet')
    parser.add_argument('--resume-from', type=int, default=None)
    parser.add_argument('--n-workers', type=int, default=32,
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
    parser.add_argument('--l2-strength', type=float, default=1e-3,
                        help='L2 regularization strength.')
    parser.add_argument('--round-steps', type=int, default=100,
                        help='#steps of training per round.')
    parser.add_argument('--test-freq', type=int, default=2, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='baseline.local',
                        help='Directory of saved results.')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    main(args)
