__version__ = '0.1.4+e2bf47f'
git_version = 'e2bf47fcaeaf94992e2e2768446cad96034b4c32'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
