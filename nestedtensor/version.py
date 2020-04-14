__version__ = '0.0.1.dev202041422+f266219'
git_version = 'f266219de6daca23be50dd0f12df0a892c96b830'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
