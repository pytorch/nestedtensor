from nestedtensor import torch
import random
import pprint

EMBED_DIM = 200

SEED = 0
def gen_tensor():
    globals()['SEED'] += 1
    return torch.tensor([globals()['SEED']])

def gen_clusters(num_clusters, size_range):

    def gen_cluster(num_entries):
        return [gen_tensor() for _ in range(num_entries)]
    
    return [gen_cluster(random.randint(*size_range)) for _ in range(num_clusters)]

clusters = gen_clusters(3, (2, 5))

# Two keys for now
keys = [gen_tensor(), gen_tensor()]
# Simulating some overlap
sub_clusters = [clusters[:3], clusters[2:]]

# Algorithm


results = []
for key in keys:
    key_results = []
    for sub_cluster in sub_clusters:
        sub_cluster_results = []
        for cluster in sub_cluster:
            sub_cluster_results.append([torch.dot(key, entry) for entry in cluster])
        key_results.append(sub_cluster_results)
    results.append(key_results)

pprint.pprint(clusters)
pprint.pprint(keys)
pprint.pprint(results)
