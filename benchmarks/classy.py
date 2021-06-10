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
        return start_event.elapsed_time(end_event) / 1e3
    else:
        return (time.time() - t0)

@torch.inference_mode()
def run_benchmark(iters, shapes, model, model_name, bsz):
    ts = []
    for s in shapes:
        inp = torch.randn(*s, dtype=torch.half).cuda()
        ts.append(inp)
    ts_nt = nestedtensor.nested_tensor([t.squeeze(0) for t in ts], device=torch.device('cuda'), dtype=torch.half)

    def _loop():
        model_outputs = []
        for inp in ts:
            model_outputs.append(model(inp))
        return model_outputs

    
    # Test
    model_outputs = _loop()
    outputs_nt = model(ts_nt)
    for mo, ntmo in zip(model_outputs, outputs_nt.unbind()):
        assert torch.allclose(mo.squeeze(0), ntmo, rtol=1e-4, atol=1e-5)

    loop_time = benchmark_torch_function(iters, _loop)
    nt_time = benchmark_torch_function(iters, lambda: model(ts_nt))

    shapes_2_array = np.array([s[2] for s in shapes])
    shapes_3_array = np.array([s[3] for s in shapes])
    print(f"model_name: {model_name.ljust(18)}", end='')
    print(f" bsz: {bsz}", end='')
    print(f" mean±std shapes[2]: {shapes_2_array.mean():.2f}±{shapes_2_array.std():.2f}", end='')
    print(f" mean±std shapes[3]: {shapes_3_array.mean():.2f}±{shapes_3_array.std():.2f}", end='')
    print(f" loop: {loop_time / iters:.2f}s, nt: {nt_time / iters:.2f}s, speedup: {loop_time / nt_time:.2f}x")

if __name__ == "__main__":
    def _benchmark(model_name):
        model = build_model({"name": model_name})
        model = model.cuda().half().eval()
        
        random.seed(123)
        shapes = [(1, 3, random.randint(100, 150), random.randint(100, 150)) for _ in range(BSZ)]
        run_benchmark(1, shapes, model, model_name, 128)
        run_benchmark(1, shapes, model, model_name, 256)

    _benchmark("resnext101_32x4d")
    _benchmark("regnet_y_128gf")

