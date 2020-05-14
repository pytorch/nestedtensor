__version__ = '0.0.1.dev202051419+185b5cf'
git_version = '185b5cfd9d01b95cb4f6f610a767521431c89f4c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
