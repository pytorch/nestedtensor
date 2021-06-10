import torch
import numpy as np
import time
import random
import nestedtensor
from classy_vision.models import build_model
model = build_model({"name": "resnext101_32x4d"})

model = model.cuda().half().eval()

ts = []
model_outputs = []
BSZ = 128
random.seed(123)
shapes = [(1, 3, random.randint(100, 150), random.randint(100, 150)) for _ in range(BSZ)]
shapes_2_array = np.array([s[2] for s in shapes])
shapes_3_array = np.array([s[3] for s in shapes])
print(f"mean/std shapes[2]: {shapes_2_array.mean():.2f}, {shapes_2_array.std():.2f}", end='')
print(f" mean/std shapes[3]: {shapes_3_array.mean():.2f}, {shapes_3_array.std():.2f}")

with torch.inference_mode():
    for s in shapes:
        inp = torch.randn(*s, dtype=torch.half).cuda()
        ts.append(inp)
    t0 = time.time()
    for inp in ts:
        model_outputs.append(model(inp))
    print(time.time() - t0)
    
    ts_nt = nestedtensor.nested_tensor([t.squeeze(0) for t in ts], device=torch.device('cuda'), dtype=torch.half)
    t0 = time.time()
    outputs_nt = model(ts_nt)
    print(time.time() - t0)
    
    for mo, ntmo in zip(model_outputs, outputs_nt.unbind()):
        assert torch.allclose(mo.squeeze(0), ntmo, rtol=1e-4, atol=1e-5)
