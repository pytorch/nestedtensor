import torch
import nestedtensor

@torch.inference_mode()
def benchmark_torch_function(iters, f, *args):
    f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

def run(bdim, embedding_dim, out_dim, min_t, max_t, iters):
    import random
    random.seed(1010)
    
    # The following is meant to emulate the lenghts of randomly sampled tokenized sentences
    lengths = [random.randint(min_t, max_t) for _ in range(bdim)]
    lengths_mean = torch.tensor(lengths, dtype=torch.float).mean().item()
    lengths_std = torch.tensor(lengths, dtype=torch.float).std().item()

    # List of sentence embeddings
    tensors = [torch.rand(i, embedding_dim).cuda() for i in lengths]
    # Create packed NestedTensor
    nt = nestedtensor.nested_tensor(tensors, device=torch.device('cuda'), dtype=torch.float)
    # Created regular padded Tensor
    data = nt.to_padded_tensor(padding=0)
    # Amount of storage used for padding only
    percentage_padded = 100 * (data.numel() - nt.numel()) / data.numel()

    # Projects embeddings into another space
    lin = torch.nn.Linear(embedding_dim, out_dim).cuda()
    nt_time = benchmark_torch_function(iters, lin, nt)
    t_time = benchmark_torch_function(iters, lin, data)

    print(f"batch size: {bdim:4.0f}, embedding dim: {embedding_dim}, out_dim: {out_dim}, T mean:{lengths_mean:5.0f}, T std: {lengths_std:4.0f}", end='')
    print(f", padding: {percentage_padded:3.0f}%, NT: {nt_time/iters:4.0f}ms, T: {t_time/iters:4.0f}ms, Speedup: {t_time/nt_time:3.2f}x")

print("CUDA device: ", torch.cuda.get_device_name(0))
iters = 10
for out_dim in [4096, 2048, 1024, 512, 256]:
    print("")
    for embed_dim in [4096, 2048, 1024, 512, 256]:
        print("")
        for min_t, max_t in [(16, 128), (32, 128), (64, 128), (128, 128)]:
            run(256, embed_dim, out_dim, min_t, max_t, iters)
