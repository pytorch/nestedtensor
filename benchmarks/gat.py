import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random
import time
import nestedtensor
from nestedtensor import nested_tensor as ntnt

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


num_features = 1433
num_classes = 7


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8,
                             dropout=0.6)

        self.conv2 = GATConv(64, num_classes, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class NTNet(torch.nn.Module):
    def __init__(self):
        super(NTNet, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8,
                             dropout=0.6)

        self.conv2 = GATConv(64, num_classes, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = ntnt([self.conv1(xi, edge_index_i) for (xi, edge_index_i) in zip(x.unbind(), edge_index.unbind())], dtype=x.dtype, device=x.device)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = ntnt([self.conv2(xi, edge_index_i) for (xi, edge_index_i) in zip(x.unbind(), edge_index.unbind())], dtype=x.dtype, device=x.device)
        return F.log_softmax(x, dim=1)


def create_models(device):
    model = Net().to(device).eval()
    nt_model = NTNet().to(device).eval()
    return model, nt_model

def create_tensors():
    random.seed(1010)
    nnodes_list = []
    nedges_list = []
    for i in range(50):
        nnodes_list.append(random.randint(100, 4000))
        nedges_list.append(random.randint(8000, 15000))
    
    tensors_x = []
    tensors_edge_index = []
    for nnodes, nedges in zip(nnodes_list, nedges_list):
        x = torch.normal(-10, 4, (nnodes, 1433))
        x[x < 0] = 0.
        x[x > 1] = 1.
        edge_index = torch.randint(0, nnodes, (2, nedges), dtype=torch.int64)
        tensors_x.append(x)
        tensors_edge_index.append(edge_index)
    return tensors_x, tensors_edge_index


@torch.inference_mode()
def loop(model, tensors_x, tensors_edge_index):
    for x, edge_index in zip(tensors_x, tensors_edge_index):
        model(x, edge_index)


@torch.inference_mode()
def nt(nt_model, nt_x, nt_edge_index):
    nt_model(nt_x, nt_edge_index)

if __name__ == "__main__":
    device = torch.device('cuda')
    model, nt_model = create_models(device)
    tensors_x, tensors_edge_index = create_tensors()
    print(benchmark_torch_function(10, loop, model, tensors_x, tensors_edge_index))
    nt_x = ntnt(tensors_x, device=device)
    nt_edge_index = ntnt(tensors_edge_index, device=device, dtype=torch.int64)
    print(benchmark_torch_function(10, nt, nt_model, nt_x, nt_edge_index))
