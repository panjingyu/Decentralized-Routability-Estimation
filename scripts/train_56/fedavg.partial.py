#!/usr/bin/env python3
"""Main script."""


import argparse
import copy
import os
import pickle
import time

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.loss import TripletMarginWithDistanceLoss
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.routability import DRVHotspotDataset
from federated.client import Client
from federated.server import fed_avg
from model.routenet import RouteNetFCN
from utils.metrics import roc_auc
from utils.regularizer import l2_regularizer


def main(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.saved = os.path.join('saved', args.saved)
    os.makedirs(args.saved, exist_ok=True)

    server = nn.DataParallel(
        RouteNetFCN(in_channels=64).to(device))

    # prepare clients
    clients = {}
    val_loaders = []
    for client in os.listdir(args.alloc_dir):
        client_dir = os.path.join(args.alloc_dir, client)
        if not os.path.isdir(client_dir):
            continue
        model = copy.deepcopy(server)
        data_train = DRVHotspotDataset(client_dir, train=True)
        data_val = DRVHotspotDataset(client_dir, train=False)
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
        optimizer = optim.SGD(model.parameters(),
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
            'device': device,
        }
        clients[client] = Client(**client_config)
        val_loaders.append(loader_val)
    client_n_batch_per_epoch = {}
    for c in sorted(clients.keys()):
        n_batch = len(clients[c].loader_train)
        print('{:}: {:2d} batchs per epoch'.format(c, n_batch))
        client_n_batch_per_epoch[c] = n_batch
    n_sync_per_epoch = min(client_n_batch_per_epoch.values())

    # fedavg
    start_time = time.time()
    loss, acc, auc = {}, {}, {}
    loss_server, acc_server, auc_server = {}, {}, {}
    for round_idx in range(args.max_round):
        print('\nRound {}/{}'.format(round_idx + 1, args.max_round))
        selected_clients = sorted(clients.keys())
        for client_idx in selected_clients:
            client = clients[client_idx]
            client.fetch_server(exclude=('layer_c4'))
            n_steps = (round_idx + 1) / n_sync_per_epoch \
                * len(client.loader_train)
            print('\nTrain {} to {:.1f} steps, {:.2f} epochs'.format(
                client_idx, n_steps, n_steps / len(client.loader_train)))
            client.train_cycled_steps(n_steps)
            if (round_idx + 1) % n_sync_per_epoch:
                continue
            print('\nValidate', client_idx)
            loss_, acc_, auc_ = client.validate_one_epoch()
            if client_idx not in loss:
                loss[client_idx] = [loss_]
                acc[client_idx] = [acc_]
                auc[client_idx] = [auc_]
            else:
                loss[client_idx].append(loss_)
                acc[client_idx].append(acc_)
                auc[client_idx].append(auc_)
        this_time = time.time()
        print('--- Elapsed time: {:.1f} sec, {:.1f} sec/round'.format(
           this_time - start_time, (this_time - start_time) / (round_idx + 1) 
        ))

        pickle.dump(loss, open(os.path.join(args.saved, 'loss.pkl'), 'wb'))
        pickle.dump(acc, open(os.path.join(args.saved, 'acc.pkl'), 'wb'))
        pickle.dump(auc, open(os.path.join(args.saved, 'auc.pkl'), 'wb'))

        # local training done, perform FedAvg
        client_states = [c.model.state_dict() for c in clients.values()]
        weight = [len(c.loader_train.dataset) for c in clients.values()]
        print('\nFed Avg')
        fed_avg(client_states, server, weight=weight)
        torch.save(server.state_dict(), os.path.join(args.saved, 'server.pt'))

        # test server
        server.eval()
        server_criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            if (round_idx + 1) % args.test_freq == 0:
                print('\nTest server at Round {}/{}'.format(
                    round_idx + 1, args.max_round))
                for client_idx in clients:
                    print('Test on', client_idx)
                    loader = clients[client_idx].loader_val
                    total = correct = 0
                    total_loss_data, total_samples = 0, 0
                    pred_epoch, target_epoch = [], []
                    for batch_idx, (data, target) in enumerate(loader):
                        data, target = data.to(device), target.to(device)
                        output = server(data)
                        prediction = output.greater(0.)
                        total += target.numel()
                        correct += prediction.eq(target).sum().item()
                        loss_data = server_criterion(output, target.float())
                        total_loss_data += loss_data.item() * target.shape[0]
                        total_samples += target.shape[0]
                        pred_epoch.append(output.cpu().numpy())
                        target_epoch.append(target.cpu().numpy())
                        if (batch_idx + 1) % 10 == 0:
                            print('[Step={:2d}] loss={:.4f}'.format(
                                batch_idx + 1, loss_data))
                    pred_epoch = np.vstack(pred_epoch)
                    target_epoch = np.vstack(target_epoch)
                    avg_loss = total_loss_data / total_samples
                    avg_acc = correct / total
                    auc_score = roc_auc(target_epoch.ravel(),
                                        pred_epoch.ravel(),
                                        device=device)
                    print('loss={:.4f} acc={:.4f} auc={:.4f}\n'.format(
                        avg_loss, avg_acc, auc_score))
                    if client_idx not in loss_server:
                        loss_server[client_idx] = [avg_loss]
                        acc_server[client_idx] = [avg_acc]
                        auc_server[client_idx] = [auc_score]
                    else:
                        loss_server[client_idx].append(avg_loss)
                        acc_server[client_idx].append(avg_acc)
                        auc_server[client_idx].append(auc_score)
        pickle.dump(loss_server,
                    open(os.path.join(args.saved, 'loss_server.pkl'), 'wb'))
        pickle.dump(acc_server,
                    open(os.path.join(args.saved, 'acc_server.pkl'), 'wb'))
        pickle.dump(auc_server,
                    open(os.path.join(args.saved, 'auc_server.pkl'), 'wb'))


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
    parser.add_argument('--l2-strength', type=float, default=1e-3,
                        help='L2 regularization strength.')
    parser.add_argument('--test-freq', type=int, default=2, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='fedavg',
                        help='Directory of saved results.')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    main(args)
