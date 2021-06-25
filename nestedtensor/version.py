__version__ = '0.1.4+5675da8'
git_version = '5675da86835134aaab9731baa15e2a514201e8b3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
