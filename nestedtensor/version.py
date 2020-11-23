__version__ = '0.0.1.dev2020112317+e70b964'
git_version = 'e70b964e17a4cdf559359f126a53106efa45cd80'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
