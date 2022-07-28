"""Client-side procedures & modules for federated optimization."""


import numpy as np

import torch
import torch.nn as nn
import torch.cuda.amp as amp

from utils.metrics import roc_auc


def _cycled(loader):
    while True:
        for item in loader:
            yield item


def fedprox_proximity(server, client):
    prox_terms = []
    if isinstance(client, nn.DataParallel):
        client = client.module
    server_state = server.state_dict()
    for name, m in client.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            server_weight = server_state[name + '.weight']
            prox_terms.append((m.weight - server_weight).pow(2).sum())
            if m.bias is not None:
                server_bias = server_state[name + '.bias']
                prox_terms.append((m.bias - server_bias).pow(2).sum())
    return sum(prox_terms)


class Client:
    """Client device class."""

    def __init__(self, model, server, loader_train, loader_val, optimizer,
                 regularizer, reg_strength, criterion, device, grad_scaler=None):
        self.model = model
        self.server = server
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_strength = reg_strength
        self.criterion = criterion
        self.device = device
        self.grad_scaler = grad_scaler

        self.loader_train_cycled = iter(self.loader_train)
        self.n_trained_cycled_steps = 0

    def train_one_epoch(self, max_steps, report_freq=10, with_amp=False):
        self.model.train()
        total = correct = 0
        for batch_idx, (data, target) in enumerate(self.loader_train):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            if with_amp:
                with amp.autocast():
                    output = self.model(data)
                    loss_data = self.criterion(output, target.float())
                    loss_reg = self.regularizer(self.model)
                    loss = loss_data + self.reg_strength * loss_reg
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
            else:
                output = self.model(data)
                loss_data = self.criterion(output, target.float())
                loss_reg = self.regularizer(self.model)
                loss = loss_data + self.reg_strength * loss_reg
                loss.backward()
                self.optimizer.step()
            prediction = output.greater(0.)
            total += target.numel()
            correct += prediction.eq(target).sum()
            if report_freq is not None and (batch_idx + 1) % report_freq == 0:
                print('[Step={:2d}] loss={:.4f} acc={:.3f}'.format(
                    batch_idx + 1, loss_data, correct / total))
            if batch_idx + 1 >= max_steps:
                break
        print('loss={:.4f} acc={:.4f}'.format(loss_data, correct / total))

    def train_cycled_steps(self, n_steps, report_freq=10):
        self.model.train()
        total = correct = 0
        # for batch_idx, (data, target) in self.loader_train_cycled:
        while True:
            self.n_trained_cycled_steps += 1
            try:
                data, target = next(self.loader_train_cycled)
            except StopIteration:
                self.loader_train_cycled = iter(self.loader_train)
                data, target = next(self.loader_train_cycled)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            prediction = output.greater(0.)
            total += target.numel()
            correct += prediction.eq(target).sum()
            loss_data = self.criterion(output, target.float())
            loss_reg = self.regularizer(self.model)
            loss = loss_data + self.reg_strength * loss_reg
            loss.backward()
            self.optimizer.step()
            if self.n_trained_cycled_steps % report_freq == 0:
                print('[Step={:2d}] loss={:.4f} acc={:.3f}'.format(
                    self.n_trained_cycled_steps, loss_data, correct / total))
            if self.n_trained_cycled_steps >= n_steps:
                break
        print('loss={:.4f} acc={:.4f}'.format(loss_data, correct / total))

    def train_fedprox_one_round(self, n_steps, fedprox_mu, report_freq=10):
        self.model.train()
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
                loss_prox = fedprox_proximity(self.server, self.model)
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

    def train_fedprox_cycled_steps(self, n_steps, fedprox_mu, report_freq=10):
        self.model.train()
        total = correct = 0
        while True:
            self.n_trained_cycled_steps += 1
            try:
                data, target = next(self.loader_train_cycled)
            except StopIteration:
                self.loader_train_cycled = iter(self.loader_train)
                data, target = next(self.loader_train_cycled)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            prediction = output.greater(0.)
            total += target.numel()
            correct += prediction.eq(target).sum()
            loss_data = self.criterion(output, target.float())
            loss_reg = self.regularizer(self.model)
            loss_prox = fedprox_proximity(self.server, self.model)
            loss = loss_data + self.reg_strength * loss_reg \
                + fedprox_mu * loss_prox
            loss.backward()
            self.optimizer.step()
            if self.n_trained_cycled_steps % report_freq == 0:
                print('[Step={:2d}] loss={:.4f} acc={:.3f}'.format(
                    self.n_trained_cycled_steps, loss_data, correct / total))
            if self.n_trained_cycled_steps >= n_steps:
                break
        print('loss={:.4f} acc={:.4f}'.format(loss_data, correct / total))

    @torch.no_grad()
    def validate_one_epoch(self, report_freq=10, loader=None, auc=True):
        if loader is None:
            loader = self.loader_val
        # dp_model = nn.DataParallel(self.model)
        # dp_model.eval()
        self.model.eval()
        total = correct = 0
        total_loss_data, total_samples = 0, 0
        pred_epoch, target_epoch = [], []
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            # output = dp_model(data)
            output = self.model(data)
            prediction = output.greater(0.)
            pred_epoch.append(output.cpu().numpy())
            target_epoch.append(target.cpu().numpy())
            total += target.numel()
            correct += prediction.eq(target).sum()
            loss_data = self.criterion(output, target.float())
            total_loss_data += loss_data.item() * target.shape[0]
            total_samples += target.shape[0]
            if report_freq is not None and (batch_idx + 1) % report_freq == 0:
                print('[Step={:2d}] loss={:.4f}'.format(
                    batch_idx + 1, loss_data))
        pred_epoch = np.vstack(pred_epoch)
        target_epoch = np.vstack(target_epoch)
        if auc:
            auc_score = roc_auc(target_epoch.ravel(), pred_epoch.ravel(),
                                device=self.device)
        avg_loss = total_loss_data / total_samples
        avg_acc = correct / total
        # del dp_model
        # torch.cuda.empty_cache()
        # np.save('pred.npy', pred_epoch)
        # np.save('target.npy', target_epoch)
        if auc:
            print('loss={:.4f} acc={:.4f} auc={:.4f}'.format(
                avg_loss, avg_acc, auc_score))
            return avg_loss, avg_acc, auc_score
        else:
            return avg_loss, avg_acc

    def fetch_server(self, excluded=(), self_weight=None):
        self.update_model(self.server,
                          excluded=excluded,
                          self_weight=self_weight)

    def update_model(self, model, excluded=(), self_weight=None):
        new_model_state = model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self_state = self.model.module.state_dict()
        else:
            self_state = self.model.state_dict()
        for key, value in new_model_state.items():
            assert key in self_state, 'Keys not match with server'
            if all(ex not in key for ex in excluded):
                if self_weight is None:
                    self_state[key] = value
                else:
                    self_state[key] = (1 - self_weight) * value \
                        + self_weight * self_state[key]
            # else:
            #     print(key, 'not fetched from server')
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(self_state)
        else:
            self.model.load_state_dict(self_state)

    def get_training_set_size(self):
        return len(self.loader_train.dataset)
