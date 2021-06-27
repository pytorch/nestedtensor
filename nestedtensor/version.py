__version__ = '0.1.4+abe8226'
git_version = 'abe8226843194532bd2bba8490ae390280e60342'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
