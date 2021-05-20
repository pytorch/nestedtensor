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
        return start_event.elapsed_time(end_event) * 1e3
    else:
        return (time.time() - t0) * 1e6


def run(bdim, embedding_dim, vocab_size, min_t, max_t, iters, device):
    import random
    random.seed(1010)

    # The following is meant to emulate the lenghts of randomly sampled tokenized sentences
    lengths = [random.randint(min_t, max_t) for _ in range(bdim)]
    lengths_mean = torch.tensor(lengths, dtype=torch.float).mean().item()
    lengths_std = torch.tensor(lengths, dtype=torch.float).std().item()

    # List of sentence embeddings
    tensors = [torch.tensor(random.randint(1, vocab_size)) for i in lengths]
    # Create packed NestedTensor
    nt = nestedtensor.nested_tensor(tensors, device=device, dtype=torch.int64)
    # Created regular padded Tensor
    data, _ = nt.to_tensor_mask()
    data = data.to(torch.int64)
    # Amount of storage used for padding only
    percentage_padded = 100 * (data.numel() - nt.numel()) / data.numel()

    # Projects embeddings into another space
    lin = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0).to(device)
    nt_time = benchmark_torch_function(iters, lin, nt)
    t_time = benchmark_torch_function(iters, lin, data)

    print(f"batch size: {bdim:4.0f}, embedding dim: {embedding_dim}, vocab_size: {vocab_size}, T mean:{lengths_mean:5.0f}, T std: {lengths_std:4.0f}", end='')
    print(f", padding: {percentage_padded:3.0f}%, NT: {nt_time/iters:4.0f}us, T: {t_time/iters:4.0f}us, Speedup: {t_time/nt_time:3.2f}x")


device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA device: ", torch.cuda.get_device_name(0))
    device = torch.device('cuda')
iters = 100
for vocab_size in [65536, 32768, 16384, 8192, 4096]:
    print("")
    for embed_dim in [4096, 2048, 1024, 512, 256]:
        print("")
        for min_t, max_t in [(16, 128), (32, 128), (64, 128), (128, 128)]:
            run(256, embed_dim, vocab_size, min_t, max_t, iters, device)
