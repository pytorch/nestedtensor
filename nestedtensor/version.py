__version__ = '0.1.4+48e80e0'
git_version = '48e80e01aae747da5539b7a246ecca45f533ce5f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
