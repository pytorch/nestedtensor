__version__ = '0.0.1.dev20201167+b1e70cc'
git_version = 'b1e70cc08a3d2aa77449f3b9929d9ef7e18b7b8d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
