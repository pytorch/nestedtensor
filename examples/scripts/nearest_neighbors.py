from nestedtensor import torch
import argparse
import time
import random
import pprint

EMBED_DIM = 256

SEED = 0


def gen_tensor():
    globals()['SEED'] += 1
    # return torch.tensor([globals()['SEED']])
    return torch.rand(EMBED_DIM)


def gen_clusters(num_clusters, size_range):

    def gen_cluster(num_entries):
        return [gen_tensor() for _ in range(num_entries)]

    return [gen_cluster(random.randint(*size_range)) for _ in range(num_clusters)]


def gen_algorithm_naive(keys, sub_clusters):
    # For-loops over vectors
    def _naive():
        results = []
        for sub_cluster, key in zip(sub_clusters, keys):
            sub_cluster_results = []
            for cluster in sub_cluster:
                sub_cluster_results.append(
                    [torch.dot(key, entry).item() for entry in cluster])
            results.append(sub_cluster_results)
        return results
    return _naive

def gen_algorithm_mv(keys, sub_clusters):
    # For-loops over vectors and matrices
    new_sub_clusters = []
    for sub_cluster in sub_clusters:
        new_sub_cluster = [torch.tensor(list(map(list, cluster))) for cluster in sub_cluster]
        new_sub_clusters.append(new_sub_cluster)
    sub_clusters = new_sub_clusters
    def _mv():
        results = []
        for sub_cluster, key in zip(sub_clusters, keys):
            sub_cluster_results = []
            for cluster in sub_cluster:
                sub_cluster_results.append(torch.mv(cluster, key))
            results.append(sub_cluster_results)
        return results
    return _mv

def gen_algorithm_nested_mv(keys, sub_clusters):
    # For-loops over vectors and matrices
    new_sub_clusters = []
    for sub_cluster in sub_clusters:
        new_sub_cluster = [torch.tensor(list(map(list, cluster))) for cluster in sub_cluster]
        new_sub_clusters.append(new_sub_cluster)
    nested_sub_clusters = torch.nested_tensor(sub_clusters).to_tensor(2)
    def _nested_mv():
        return torch.mv(nested_sub_clusters, torch.nested_tensor(keys))
    return _nested_mv


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

def benchmark_fn(fn, run_time = 5.0):
    times = []
    num_runs = 0
    t = 0.0
    while (t < run_time):
        ti = time.time()
        fn()
        ti = time.time() - ti
        t += ti
        times.append(ti)
    times = torch.tensor(times) * 1e6
    return "fn {:<15} avg(us): {:10.4f} std(us): {:10.4f} num_runs: {}".format(fn.__name__, times.mean().item(), times.std().item(), len(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-results', dest='print_results', action='store_true')
    args = parser.parse_args()
    clusters = gen_clusters(3, (2, 5))

    # Two keys for now
    keys = [gen_tensor(), gen_tensor()]
    # Simulating some overlap
    sub_clusters = [clusters[:3], clusters[2:]]

    # Get algorithm
    gen_results_naive = gen_algorithm_naive(keys, sub_clusters)
    gen_results_mv = gen_algorithm_mv(keys, sub_clusters)
    gen_results_nested_mv = gen_algorithm_nested_mv(keys, sub_clusters)

    # print(benchmark_fn(gen_results_naive))
    # print(benchmark_fn(gen_results_mv))
    import profile
    profile.runctx('benchmark_fn(gen_results_nested_mv)', globals, locals)
    # print(benchmark_fn(gen_results_nested_mv))

    if args.print_results:
        print('naive')
        print_results(gen_results_naive(), keys, sub_clusters)
        print('\nmv')
        print_results(gen_results_mv(), keys, sub_clusters)
        print('\nnested_mv')
        print_results(gen_results_nested_mv(), keys, sub_clusters)
