__version__ = '0.1.4+4681334'
git_version = '46813348a62a77fe0ce7288a44615d123dd38bc4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
