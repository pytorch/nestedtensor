__version__ = '0.1.4+0717f49'
git_version = '0717f4923d08fd5985074f35c31f9be856b7a8f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
