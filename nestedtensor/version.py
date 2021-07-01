__version__ = '0.1.4+0e27cd0'
git_version = '0e27cd0c47aceb846359b28b991091b531432f7a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
