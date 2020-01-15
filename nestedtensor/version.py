__version__ = '0.0.1.dev202011421+1f8c400'
git_version = '1f8c400ce056507c6cd002a528bf8f88c7c59ad2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
