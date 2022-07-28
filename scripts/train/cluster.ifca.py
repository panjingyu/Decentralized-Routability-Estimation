#!/usr/bin/env python3
"""Main script."""


import argparse
import copy
import json
import os
import pickle
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.routability import DRVHotspotDataset
from federated.client.ifca import IFCAClient
from federated.server.ifca import ifca_avg
from model.routenet import RouteNetFCN
from model.custom import CompactCNN2
from model.routenet_56 import RouteNetFCN56
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

    in_channels = args.n_channels
    label_size = args.label_size

    if args.model == 'routenet224':
        if args.id_init:
            cluster_models = [RouteNetFCN(in_channels=in_channels)]
            for _ in range(1, args.n_clusters):
                cluster_models.append(copy.deepcopy(cluster_models[0]))
        else:
            cluster_models = [RouteNetFCN(in_channels=in_channels)
                              for _ in range(args.n_clusters)]
    elif args.model == 'routenet56':
        if args.id_init:
            cluster_models = [RouteNetFCN56(in_channels=in_channels)]
            for _ in range(1, args.n_clusters):
                cluster_models.append(copy.deepcopy(cluster_models[0]))
        else:
            cluster_models = [RouteNetFCN56(in_channels=in_channels)
                              for _ in range(args.n_clusters)]
    elif args.model == 'pros224':
        if args.id_init:
            cluster_models = [PROS(in_channels=in_channels)]
            for _ in range(1, args.n_clusters):
                cluster_models.append(copy.deepcopy(cluster_models[0]))
        else:
            cluster_models = [PROS(in_channels=in_channels)
                              for _ in range(args.n_clusters)]
    elif args.model == 'compactcnn2':
        if args.id_init:
            cluster_models = [CompactCNN2(in_channels=in_channels)]
            for _ in range(1, args.n_clusters):
                cluster_models.append(copy.deepcopy(cluster_models[0]))
        else:
            cluster_models = [CompactCNN2(in_channels=in_channels)
                              for _ in range(args.n_clusters)]
    else:
        raise NotImplementedError

    use_assigned_cluster = args.cluster_config is not None
    if use_assigned_cluster:
        with open(args.cluster_config, 'r') as f:
            cluster_config = json.load(f)
            print('Cluster assignment:', cluster_config)
        assert args.random_init_cluster is False

    use_partial = args.local_layers_config is not None
    if use_partial:
        with open(args.local_layers_config, 'r') as f:
            local_layers = json.load(f)[args.model]
            print('Local layers:', local_layers)
    else:
        local_layers = ()

    # prepare clients
    clients = {}
    val_loaders = []
    client_cnt = 0
    for client in sorted(os.listdir(args.alloc_dir)):
        client_dir = os.path.join(args.alloc_dir, client)
        if not os.path.isdir(client_dir):
            continue
        model = copy.deepcopy(cluster_models[0])
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
        data_train = DRVHotspotDataset(client_dir, train=True,
                                       n_channels=in_channels, label_size=label_size)
        data_val = DRVHotspotDataset(client_dir, train=False,
                                     n_channels=in_channels, label_size=label_size)
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
            'server': cluster_models,
            'loader_train': loader_train,
            'loader_val': loader_val,
            'optimizer': optimizer,
            'regularizer': l2_regularizer,
            'reg_strength': args.l2_strength,
            'criterion': nn.BCEWithLogitsLoss(),
            'device': torch.device(client_device),
            'cluster_id': None,
        }
        clients[client] = IFCAClient(**client_config)
        val_loaders.append(loader_val)
        client_cnt += 1

    client_keys = sorted(clients.keys())
    clients_weight = [clients[c].get_training_set_size() for c in client_keys]

    if use_assigned_cluster:
        assert args.n_clusters - 1 == max([cluster_config[c] for c in client_keys])


    # IFCA
    loss, acc, auc = {}, {}, {}
    loss_server, acc_server, auc_server = {}, {}, {}
    start = time.time()
    for round_idx in range(args.max_round):
        print('\n[Round {}/{}]'.format(round_idx + 1, args.max_round))
        selected_clients = client_keys # Select all clients in each round
        for client_idx in selected_clients:
            client = clients[client_idx]
            client.model.to(client.device)
            print('Train', client_idx, end=': ')
            if use_assigned_cluster:
                client.fetch_server(assign_cluster=cluster_config[client_idx],
                                    excluded=local_layers)
            elif round_idx == 0 and args.random_init_cluster:
                client.fetch_server(random_cluster=True,
                                    excluded=local_layers)
            elif args.self_avg_weight is not None:
                print('Fetched as %.2f of all clients' % args.self_avg_weight)
                self_weight_on_fedavg = \
                    client.get_training_set_size() / sum(clients_weight)
                self_weight_on_fetch = \
                    args.self_avg_weight / (1 - self_weight_on_fedavg)
                client.fetch_server(self_weight=self_weight_on_fetch,
                                    excluded=local_layers)
            else:
                client.fetch_server(excluded=local_layers)
            client.train_fedprox_one_round(args.round_steps, args.fedprox_mu,
                                           report_freq=None)
            if (round_idx + 1) % args.val_freq != 0:
                client.model.to('cpu')
                continue
            print('Val: ', end='')
            loss_, acc_, auc_ = client.validate_one_epoch(report_freq=None)
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
        print('\nIFCA Avg')
        ifca_avg(clients, cluster_models, names_not_merge=local_layers,
                 verbose=use_partial)
        torch.save(cluster_models,
                   os.path.join(args.saved, 'server-{}.pt'.format(round_idx+1)))

        # test server
        server_criterion = nn.BCEWithLogitsLoss()
        for idx, server in enumerate(cluster_models):
            with torch.no_grad():
                if (round_idx + 1) % args.test_freq == 0:
                    server.to(device)
                    server.eval()
                    print('\nTest cluster {} at Round {}/{}'.format(
                        idx, round_idx + 1, args.max_round))
                    for client_idx in client_keys:
                        print('Test on', client_idx, end=': ')
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
                            # if (batch_idx + 1) % 10 == 0:
                            #     print('[Step={:2d}] loss={:.4f}'.format(
                            #         batch_idx + 1, loss_data))
                        pred_epoch = np.vstack(pred_epoch)
                        target_epoch = np.vstack(target_epoch)
                        avg_loss = total_loss_data / total_samples
                        avg_acc = correct / total
                        auc_score = roc_auc(target_epoch.ravel(),
                                            pred_epoch.ravel(),
                                            device=device)
                        print('loss={:.4f} acc={:.4f} auc={:.4f}'.format(
                            avg_loss, avg_acc, auc_score))
                        if client_idx not in loss_server:
                            loss_server[client_idx] = [avg_loss]
                            acc_server[client_idx] = [avg_acc]
                            auc_server[client_idx] = [auc_score]
                        else:
                            loss_server[client_idx].append(avg_loss)
                            acc_server[client_idx].append(avg_acc)
                            auc_server[client_idx].append(auc_score)
                    server = server.to('cpu')
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
    parser.add_argument('--n-channels', type=int, default=64,
                        help='Number of channels in input features.')
    parser.add_argument('--max-round', type=int, default=100,
                        help='Maximum number of rounds.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training & validation.')
    parser.add_argument('--l2-strength', type=float, default=1e-3,
                        help='L2 regularization strength.')
    parser.add_argument('--fedprox-mu', type=float, default=0,
                        help='FedProx proximity term strength.')
    parser.add_argument('--self-avg-weight', type=float, default=None,
                        help='Weight on self parameters when performing avg.')
    parser.add_argument('--round-steps', type=int, default=100,
                        help='#steps of training per round.')
    parser.add_argument('--val-freq', type=int, default=5, metavar='N',
                        help='Validate clients every N rounds.')
    parser.add_argument('--test-freq', type=int, default=5, metavar='N',
                        help='Test server every N rounds.')
    parser.add_argument('--saved', type=str, default='ifca',
                        help='Directory of saved results.')
    parser.add_argument('--model', type=str, default='routenet')
    parser.add_argument('--label-size', type=int, default=56,
                        help='Length of label map.')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of clusters')
    parser.add_argument('--id-init', action='store_true',
                        help='Use identical init on cluster models')
    parser.add_argument('--random-init-cluster', action='store_true',
                        help='Clients adpot random clusters at the 1st round')
    parser.add_argument('--cluster-config', type=str, default=None,
                        help='Config json file assigning cluster to clients explicitly')
    parser.add_argument('--local-layers-config', type=str, default=None,
                        help='Global-local config.')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    start = time.time()
    main(args)
    end = time.time()
    print('=== Total run time: {:.2f} hrs ==='.format((end - start) / 3600))
