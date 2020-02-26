import torch 

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
mask = torch.tensor([True, True, False])

print("mask dim: ", mask.dim())

#res = a.masked_select(mask)
#print(res.size())
#tensors = [a[i] if mask[i]                 else None for i in range(len(mask))] 
#print(res)

#b = torch.reshape(res, a.size())
#print(b)

for (t, m) in zip(tensor, mask):
    print(t)
    print(m)

    print("\n")
    