__version__ = '0.1.4+3692f6f'
git_version = '3692f6fe40cbd129b694299e6728e83721d1f01e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
