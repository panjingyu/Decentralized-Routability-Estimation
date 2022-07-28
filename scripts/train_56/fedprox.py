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
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.routability import DRVHotspotDataset
from federated.client import Client
from federated.server import fed_avg
from model.custom_56 import CompactCNN56
from model.routenet import RouteNetFCN
from model.pros import PROS
from utils.metrics import roc_auc
from utils.regularizer import l2_regularizer


def main(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_device_cnt = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        cuda_device_cnt = None

    args.saved = os.path.join('saved', args.saved)
    os.makedirs(args.saved, exist_ok=True)

    if args.model == 'routenet':
        server = RouteNetFCN(in_channels=64)
    elif args.model == 'pros':
        server = PROS(in_channels=64)
    elif args.model == 'compactcnn56':
        server = CompactCNN56(in_channels=args.n_channels)
    else:
        raise NotImplementedError

    # prepare clients
    clients = {}
    val_loaders = []
    client_cnt = 0
    for client in os.listdir(args.alloc_dir):
        client_dir = os.path.join(args.alloc_dir, client)
        if not os.path.isdir(client_dir):
            continue
        model = copy.deepcopy(server)
        if cuda_device_cnt is not None:
            main_id = client_cnt % cuda_device_cnt
            client_device = torch.device('cuda:{}'.format(main_id))
            device_ids = list(range(cuda_device_cnt))
            if main_id != 0:
                device_ids[0], device_ids[main_id] = main_id, 0
            print(client, device_ids)
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            client_device = device
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
            'device': torch.device(client_device),
        }
        clients[client] = Client(**client_config)
        val_loaders.append(loader_val)
        client_cnt += 1

    client_keys = sorted(clients.keys())
    clients_weight = [clients[c].get_training_set_size() for c in client_keys]


    # fedprox
    loss, acc, auc = {}, {}, {}
    loss_server, acc_server, auc_server = {}, {}, {}
    start = time.time()
    for round_idx in range(args.max_round):
        print('\nRound {}/{}'.format(round_idx + 1, args.max_round))
        selected_clients = client_keys
        for client_idx in selected_clients:
            client = clients[client_idx]
            client.model.to(client.device)
            server.to(client.device)
            if args.self_avg_weight is not None:
                print('Fetched as %.2f of all clients' % args.self_avg_weight)
                self_weight_on_fedavg = \
                    client.get_training_set_size() / sum(clients_weight)
                self_weight_on_fetch = \
                    args.self_avg_weight / (1 - self_weight_on_fedavg)
                client.fetch_server(self_weight=self_weight_on_fetch)
            else:
                client.fetch_server()
            print('\nTraining', client_idx)
            client.train_fedprox_one_round(args.round_steps, args.fedprox_mu)
            print('\nValidation')
            loss_, acc_, auc_ = client.validate_one_epoch()
            client.model.to('cpu')
            # print(torch.cuda.memory_summary())
            if client_idx not in loss:
                loss[client_idx] = [loss_]
                acc[client_idx] = [acc_]
                auc[client_idx] = [auc_]
            else:
                loss[client_idx].append(loss_)
                acc[client_idx].append(acc_)
                auc[client_idx].append(auc_)

        pickle.dump(loss, open(os.path.join(args.saved, 'loss.pkl'), 'wb'))
        pickle.dump(acc, open(os.path.join(args.saved, 'acc.pkl'), 'wb'))
        pickle.dump(auc, open(os.path.join(args.saved, 'auc.pkl'), 'wb'))

        # local training done, perform FedAvg
        server.to('cpu')
        client_states = [clients[c].model.module.state_dict() for c in client_keys]
        print('\nFed Avg')
        fed_avg(client_states, server, weight=clients_weight)
        torch.save(server.state_dict(), os.path.join(args.saved, 'server-{}.pt'.format(round_idx+1)))

        # test server
        server_criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            if (round_idx + 1) % args.test_freq == 0:
                server.to(device)
                server.eval()
                print('\nTest server at Round {}/{}'.format(
                    round_idx + 1, args.max_round))
                for client_idx in client_keys:
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

        end = time.time()
        print('this round\'s elapsed time: {:.1f} sec'.format(end - start))
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
    parser.add_argument('--max-round', type=int, default=20,
                        help='Maximum number of rounds.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training & validation.')
    parser.add_argument('--l2-strength', type=float, default=1e-3,
                        help='L2 regularization strength.')
    parser.add_argument('--fedprox-mu', type=float, default=1e-2,
                        help='FedProx proximity term strength.')
    parser.add_argument('--self-avg-weight', type=float, default=None)
    parser.add_argument('--round-steps', type=int, default=100,
                        help='#steps of training per round.')
    parser.add_argument('--test-freq', type=int, default=2, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='fedprox',
                        help='Directory of saved results.')
    parser.add_argument('--model', type=str, default='routenet')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    main(args)
