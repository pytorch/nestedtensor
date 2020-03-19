from nestedtensor import torch
import time
import random
import pprint

import cProfile, pstats, io
from pstats import SortKey

EMBED_DIM = 256

SEED = 0


def gen_tensor():
    globals()['SEED'] += 1
    # return torch.tensor([globals()['SEED']])
    return torch.rand(EMBED_DIM)

def benchmark_fn(fn, run_time = 5.0, use_cprofile=False, warmup=1.0):
    times = []
    t = 0.0
    pr = cProfile.Profile()
    cuda_avail = torch.cuda.is_available()
    while (t < run_time):
        ti = time.time()
        if use_cprofile:
            pr.enable()
        fn()
        if cuda_avail:
            torch.cuda.synchronize()
        if use_cprofile:
            pr.disable()
        ti = time.time() - ti
        t += ti
        if warmup is not None:
            if t > warmup:
                warmup = None
                t = 0
            continue
        times.append(ti)
    times = torch.tensor(times) * 1e6
    if use_cprofile:
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
    result = ""
    result +=  "fn {:<15} avg(us): {:10.4f} std(us): {:10.4f} num_runs: {}".format(fn.__name__, times.mean().item(), times.std().item(), len(times))
    if use_cprofile:
        result += s.getvalue()
    return result
