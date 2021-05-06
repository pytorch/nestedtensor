__version__ = '0.1.4+022f325'
git_version = '022f325ae5cd0db490b38362f473d81b39c4e179'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
