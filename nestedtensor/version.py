__version__ = '0.0.1.dev2020112319+30c0843'
git_version = '30c0843ee03e6544cba67b86c63fbe17a8359ee0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
