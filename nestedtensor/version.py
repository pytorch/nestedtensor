__version__ = '0.0.1.dev20201822+c1f8ab5'
git_version = 'c1f8ab5cf44081b660d3c239b120699d349ee752'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
