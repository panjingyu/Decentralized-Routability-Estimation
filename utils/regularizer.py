"""Regularizer Tools."""


import torch.nn as nn


def l2_regularizer(model):
    reg_terms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            reg_terms.append(m.weight.pow(2).sum())
    return sum(reg_terms)
