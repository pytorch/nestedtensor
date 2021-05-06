__version__ = '0.1.4+32174cc'
git_version = '32174cc8a19222b7d143913730ccd9ef54281ca5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
