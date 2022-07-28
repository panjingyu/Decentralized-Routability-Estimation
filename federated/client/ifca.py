"""Client class with support of Iterative Federated Clustering Algorithm (IFCA)."""

import numpy as np
import torch

from federated.client import Client
from federated.client.base import fedprox_proximity


class IFCAClient(Client):
    """Client device class that supports IFCA."""

    def __init__(self, model, server, loader_train, loader_val, optimizer,
                 regularizer, reg_strength, criterion, device, cluster_id):
        self.cluster_id = cluster_id
        # self.server should be a list of clustering models
        super().__init__(model, server, loader_train, loader_val, optimizer,
                         regularizer, reg_strength, criterion, device)

    @torch.no_grad()
    def fetch_server(self, excluded=(), self_weight=None, assign_cluster=None, random_cluster=False):
        if random_cluster and assign_cluster is None:
            self.cluster_id = np.random.randint(len(self.server))
            print("Randomly adopt cluster", self.cluster_id, end='; ')
        else:
            if assign_cluster is not None:
                self.cluster_id = assign_cluster
                print("Assigned cluster ", self.cluster_id, end='; ')
            else:
                losses = []
                for model in self.server:
                    self.update_model(model)
                    loss, acc= self.validate_one_epoch(report_freq=None, auc=False)
                    losses.append(loss)
                print("Losses={}".format(np.around(losses, decimals=3)), end=' -> ')
                self.cluster_id = np.argmin(losses)
            print("Adopt cluster", self.cluster_id, end='; ')
        self.update_model(self.server[self.cluster_id],
                        excluded=excluded,
                        self_weight=self_weight)

    def train_fedprox_one_round(self, n_steps, fedprox_mu, report_freq=10):
        self.model.train()
        server = self.server[self.cluster_id].to(self.device)
        total = correct = 0
        for batch_idx, (data, target) in enumerate(self.loader_train):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            prediction = output.greater(0.)
            total += target.numel()
            correct += prediction.eq(target).sum()
            loss_data = self.criterion(output, target.float())
            loss_reg = self.regularizer(self.model)
            if fedprox_mu != 0:
                loss_prox = fedprox_proximity(server, self.model)
            else:
                loss_prox = 0.
            loss = loss_data + self.reg_strength * loss_reg \
                   + fedprox_mu * loss_prox
            loss.backward()
            self.optimizer.step()
            if report_freq is not None and (batch_idx + 1) % report_freq == 0:
                print('[Step={:2d}] loss={:.4f} acc={:.3f}'.format(
                    batch_idx + 1, loss_data, correct / total))
            if batch_idx + 1 >= n_steps:
                break
        print('loss={:.4f} acc={:.4f}'.format(loss_data, correct / total))
        self.server[self.cluster_id].to('cpu')