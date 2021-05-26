__version__ = '0.1.4+e1d384f'
git_version = 'e1d384fea9d70a664b38a53768f82c81057a7d13'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
