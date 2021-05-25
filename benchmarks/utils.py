import torch
import time

import cProfile
import pstats
import io
from pstats import SortKey

EMBED_DIM = 256

SEED = 0


def gen_tensor():
    globals()['SEED'] += 1
    # return torch.tensor([globals()['SEED']])
    return torch.rand(EMBED_DIM)


def benchmark_fn(fn, run_time=5.0, use_cprofile=False, warmup=1.0, cuda=False):
    times = []
    t = 0.0
    pr = cProfile.Profile()
    while (t < run_time):
        if cuda:
            torch.cuda.synchronize()
        ti = time.perf_counter()
        if use_cprofile:
            pr.enable()
        fn()
        if cuda:
            torch.cuda.synchronize()
        if use_cprofile:
            pr.disable()
        ti = time.perf_counter() - ti
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
    result = {}
    result['name'] = fn.__name__
    result['avg_us'] = times.mean().item()
    result['std_us'] = times.std().item()
    result['runs'] = len(times)
    if use_cprofile:
        result['cprofile'] = s.getvalue()
    return result
