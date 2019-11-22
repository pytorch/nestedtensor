from nestedtensor import torch
import random
import pprint

EMBED_DIM = 200

SEED = 0


def gen_tensor():
    globals()['SEED'] += 1
    # return torch.tensor([globals()['SEED']])
    return torch.rand(EMBED_DIM)


def gen_clusters(num_clusters, size_range):

    def gen_cluster(num_entries):
        return [gen_tensor() for _ in range(num_entries)]

    return [gen_cluster(random.randint(*size_range)) for _ in range(num_clusters)]


def gen_algorithm(keys, sub_clusters):
    def _algorithm():
        results = []
        for sub_cluster, key in zip(sub_clusters, keys):
            sub_cluster_results = []
            for cluster in sub_cluster:
                sub_cluster_results.append(
                    [torch.dot(key, entry).item() for entry in cluster])
            results.append(sub_cluster_results)
        return results
    return _algorithm


def print_results(results, keys, sub_clusters, print_details=False):
    if print_details:
        for i, sub_cluster in enumerate(sub_clusters):
            print("\n\u001b[31msub cluster {} count {} total number of entries {}\u001b[0m".format(
                i, len(sub_cluster), sum(map(len, sub_cluster))))
            pprint.pprint(sub_cluster)
        print("\nkeys")
        pprint.pprint(keys)
        print("")

    for i, result in enumerate(results):
        print(
            "result scores for \u001b[31msub cluster {} and key {}\u001b[0m".format(i, i))
        pprint.pprint(result)


if __name__ == "__main__":
    clusters = gen_clusters(3, (2, 5))

    # Two keys for now
    keys = [gen_tensor(), gen_tensor()]
    # Simulating some overlap
    sub_clusters = [clusters[:3], clusters[2:]]

    # Get algorithm
    gen_results = gen_algorithm(keys, sub_clusters)

    print_results(gen_results(), keys, sub_clusters)
