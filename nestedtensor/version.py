__version__ = '0.0.1.dev202051518+0ad410c'
git_version = '0ad410c912dec5197b7440a77e6bfbfe67609210'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
