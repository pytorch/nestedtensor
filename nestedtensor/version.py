__version__ = '0.0.1.dev2019102815+351a6b8'
git_version = '351a6b8aa271c37dd0aad98fa31e06efad21912f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
