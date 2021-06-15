import torch
import time
import nestedtensor


@torch.inference_mode()
def benchmark_torch_function(iters, f, *args):
    f(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()
    for _ in range(iters):
        f(*args)
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    else:
        return (time.time() - t0) * 1e3


# def run(bdim, embedding_dim, out_dim, min_t, max_t, iters, device):
def run(bdim, nchannel, min_t, max_t, iters, device):
    import random
    random.seed(1010)

    # The following is meant to emulate the lenghts of randomly sampled tokenized sentences
    lengths1 = [random.randint(min_t, max_t) for _ in range(bdim)]
    lengths2 = [random.randint(min_t, max_t) for _ in range(bdim)]

    # List of sentence embeddings
    tensors = [torch.rand(nchannel, l1, l2).to(device=device, dtype=torch.float) for (l1, l2) in zip(lengths1, lengths2)]
    # Create packed NestedTensor
    nt = nestedtensor.nested_tensor(tensors, device=device, dtype=torch.float)

    lin = torch.nn.Conv2d(nchannel, nchannel, (1, 1), bias=False).to(device)

    def _loop(tensors):
        result = []
        for t in tensors:
            result.append(lin(t.unsqueeze(0)).squeeze(0))
        return result

    nt_time = benchmark_torch_function(iters, lin, nt)
    t_time = benchmark_torch_function(iters, _loop, tensors)

    # print(f"batch size: {bdim:4.0f}, embedding dim: {embedding_dim}, out_dim: {out_dim}, T mean:{lengths_mean:5.0f}, T std: {lengths_std:4.0f}", end='')
    print(f"batch size: {bdim:4.0f}, nchannel: {nchannel:4.0f}", end='')
    # print(f", padding: {percentage_padded:3.0f}%, NT: {nt_time/iters:4.0f}ms, T: {t_time/iters:4.0f}ms, Speedup: {t_time/nt_time:3.2f}x")
    print(f", NT: {nt_time/iters:4.0f}ms, T: {t_time/iters:4.0f}ms, Speedup: {t_time/nt_time:3.2f}x")


if torch.cuda.is_available():
    print("CUDA device: ", torch.cuda.get_device_name(0))
iters = 10
for nchannel in [3, 128, 256, 512]:
    for min_t, max_t in [(16, 128), (32, 128), (64, 128), (128, 128)]:
        run(256, nchannel, min_t, max_t, iters, torch.device('cuda'))
        break
