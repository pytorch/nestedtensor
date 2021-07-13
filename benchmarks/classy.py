import torch
import numpy as np
import time
import random
import nestedtensor
from classy_vision.models import build_model


@torch.inference_mode()
def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()
    for _ in range(iters):
        f(*args, **kwargs)
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    else:
        return (time.time() - t0)


@torch.inference_mode()
def run_benchmark(iters, shapes, model, model_name, bsz):
    ts = []
    for s in shapes:
        inp = torch.randn(*s, dtype=torch.half).cuda()
        ts.append(inp)
    ts_nt = nestedtensor.nested_tensor([t.squeeze(0) for t in ts], device=torch.device('cuda'), dtype=torch.half)
    ts_padded = ts_nt.to_padded_tensor()
    ts_nt = nestedtensor.nested_tensor([t.squeeze(0) for t in ts], device=torch.device('cuda'), dtype=torch.half, channels_last=True)

    def _loop():
        model_outputs = []
        for inp in ts:
            model_outputs.append(model(inp))
        return model_outputs

    def _padded():
        return model(ts_padded)

    # Test
    outputs_nt = model(ts_nt)
    # import time; time.sleep(1)
    # outputs_nt = model(ts_nt)
    # import sys; sys.exit(1)
    model_outputs = _loop()
    for mo, ntmo in zip(model_outputs, outputs_nt.unbind()):
        # Using float16 tolerances from torch/testing/_core.yp
        assert torch.allclose(mo.squeeze(0), ntmo, rtol=1e-3, atol=1e-3)

    loop_time = benchmark_torch_function(iters, _loop)
    padded_time = benchmark_torch_function(iters, _padded)
    nt_time = benchmark_torch_function(iters, lambda: model(ts_nt))

    shapes_2_array = np.array([s[2] for s in shapes])
    shapes_3_array = np.array([s[3] for s in shapes])
    print(f"model_name: {model_name.rjust(18)},", end='')
    print(f" bsz: {bsz:3.0f},", end='')
    print(f" mean±std shapes[2]: {shapes_2_array.mean():.2f}±{shapes_2_array.std():.2f},", end='')
    print(f" mean±std shapes[3]: {shapes_3_array.mean():.2f}±{shapes_3_array.std():.2f},", end='')
    print(f" padded_size: {tuple(ts_padded.size())},", end='')
    print(f" loop: {loop_time / iters:7.2f}ms, nt: {nt_time / iters:7.2f}ms, padded: {padded_time / iters:7.2f}ms, speedup: {loop_time / nt_time:.2f}x")

if __name__ == "__main__":
    iters = 10

    def _benchmark(model_name, bsz):
        model = build_model({"name": model_name})
        model = model.cuda().half().eval()
        random.seed(123)
        shapes = [(1, 3, random.randint(100, 600), random.randint(100, 600)) for _ in range(bsz)]
        run_benchmark(iters, shapes, model, model_name, bsz)

    for bsz in [16, 32, 64, 128]:
        _benchmark("resnext101_32x4d", bsz)

    for bsz in [16, 32]:
        _benchmark("regnet_y_128gf", bsz)
