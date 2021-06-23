__version__ = '0.1.4+1417c1f'
git_version = '1417c1fea88bd969d3a124dee06f665e0af75c1a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
