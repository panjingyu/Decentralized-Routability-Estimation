"""Server-side procedures & modules for federated optimization."""


import torch
import torch.nn as nn


def belongs_layer_of_name(weight_key, layer_name):
    return any(layer_name == key_sp for key_sp in weight_key.split('.'))

@torch.no_grad()
def fed_avg(client_states, server_model, names_not_merge=(),
            weight=None, verbose=False):
    num_clients = len(client_states)
    if weight is None:
        weight = torch.ones(num_clients)
    else:
        weight = torch.FloatTensor(weight)
    total_weight = weight.sum()
    merged_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)
    for name, m in server_model.named_modules():
        if any(belongs_layer_of_name(name, n) for n in names_not_merge):
            if verbose:
                print('Skip layer:', name)
            continue
        if isinstance(m, merged_modules):
            if verbose:
                print('Merging layer:', name)
            m_data_avg = torch.zeros_like(m.weight)
            for c, c_state in enumerate(client_states):
                m_data_avg.add_(c_state[name+'.weight'].cpu(), alpha=weight[c])
            m_data_avg.div_(total_weight)
            m.weight.copy_(m_data_avg)
            if m.bias is not None:
                m_data_avg = torch.zeros_like(m.bias)
                for c, c_state in enumerate(client_states):
                    m_data_avg.add_(c_state[name+'.bias'].cpu(), alpha=weight[c])
                m_data_avg.div_(total_weight)
                m.bias.copy_(m_data_avg)
