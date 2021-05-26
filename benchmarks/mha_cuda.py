import torch
import time
import nestedtensor


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
        return start_event.elapsed_time(end_event) * 1e3
    else:
        return (time.time() - t0) * 1e6


def run(bdim, embedding_dim, nhead, min_t, max_t, iters, device):
    import random
    random.seed(1010)

    # The following is meant to emulate the lenghts of randomly sampled tokenized sentences
    lengths = [random.randint(min_t, max_t) for _ in range(bdim)]
    lengths_mean = torch.tensor(lengths, dtype=torch.float).mean().item()
    lengths_std = torch.tensor(lengths, dtype=torch.float).std().item()

    # List of sentence embeddings
    tensors = [torch.rand(i, embedding_dim) for i in lengths]
    # Create packed NestedTensor
    nt = nestedtensor.nested_tensor(tensors, device=device, dtype=torch.float)

    # Create MHA with self-attention in mind
    mha = torch.nn.MultiheadAttention(embedding_dim, nhead).to(device).eval()

    # Create regular padded Tensor with corresponding mask
    data, mask = nt.to_tensor_mask(mask_dim=2)
    # Prepare input for torch.nn.MHA, which is batch second for Tensor input
    data = data.transpose(0, 1)
    not_mask = torch.logical_not(mask)

    # Comparison test to show correctness and API differences
    with torch.inference_mode():
        nt_output, _ = mha(nt, nt, nt, need_weights=False)
        t_output, _ = mha(data, data, data, key_padding_mask=not_mask, need_weights=False)
        nt_output_padded = nt_output.to_padded_tensor(padding=0)
        t_output = t_output.transpose(0, 1)
        # Fill in zero for masked-out values to enable comparison
        t_output = t_output * mask.unsqueeze(-1)
        # Tolerances taken from torch/testing/_core.py
        assert torch.isclose(nt_output_padded, t_output, rtol=1e-4, atol=1e-5).all().item()

    # Time NT version
    nt_time = benchmark_torch_function(iters, mha, nt, nt, nt, need_weights=False)

    # Amount of storage used for padding only
    percentage_padded = 100 * (data.numel() - nt.numel()) / data.numel()

    # Time Tensor version
    t_time = benchmark_torch_function(iters, mha, data, data, data, key_padding_mask=not_mask, need_weights=False)

    print(f"batch size: {bdim:4.0f}, embedding dim: {embedding_dim}, nhead: {nhead}, T mean:{lengths_mean:5.0f}, T std: {lengths_std:4.0f}", end='')
    print(f", padding: {percentage_padded:3.0f}%, NT: {nt_time/iters:4.0f}us, T: {t_time/iters:4.0f}us, Speedup: {t_time/nt_time:3.2f}x")


device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA device: ", torch.cuda.get_device_name(0))
    device = torch.device('cuda')
iters = 10
for nhead in [2, 4, 8]:
    print("")
    for embed_dim in [1024, 512, 256, 128]:
        print("")
        for min_t, max_t in [(16, 128), (32, 128), (64, 128), (128, 128)]:
            run(256, embed_dim, nhead, min_t, max_t, iters, device)
