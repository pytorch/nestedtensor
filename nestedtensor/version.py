__version__ = '0.1.4+6b987e6'
git_version = '6b987e64cba95e468f2334e2920cc74d13096c68'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
