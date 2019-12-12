__version__ = '0.0.1.dev2019121219+a7f1dab'
git_version = 'a7f1dab22ab61bd1f34b0386118eea70fc1e9ba0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
