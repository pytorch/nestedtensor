__version__ = '0.1.4+e574626'
git_version = 'e574626f826c6e062baac889e5f1a333ffeacf9f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
