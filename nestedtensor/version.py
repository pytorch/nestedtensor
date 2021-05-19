__version__ = '0.1.4+3590e2e'
git_version = '3590e2ea7d78aa8579e4665f304bf406b89ae4a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
