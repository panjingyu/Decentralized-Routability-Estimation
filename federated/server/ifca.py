"""IFCA averaging."""


import torch

from federated.server import fed_avg


@torch.no_grad()
def ifca_avg(clients, cluster_models, names_not_merge=(), verbose=False):
    n_clusters = len(cluster_models)
    clustered_clients = {}
    for cluster_id in range(n_clusters):
        clustered_clients[cluster_id] = []
        for c in clients:
            if clients[c].cluster_id == cluster_id:
                clustered_clients[cluster_id].append(c)
    for cluster_id in clustered_clients:
        cl_clients = clustered_clients[cluster_id]
        if len(cl_clients) > 0:
            client_states = [clients[c].model.module.state_dict()
                             for c in cl_clients]
            fed_avg(client_states, cluster_models[cluster_id],
                    names_not_merge=names_not_merge,
                    verbose=verbose)
            print("Cluster {}: avg of {} clients: {}".format(
                cluster_id, len(cl_clients), sorted(cl_clients)
            ))
        else:
            print("Cluster {} has no clients!".format(cluster_id))
    