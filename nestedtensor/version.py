__version__ = '0.1.4+65036c3'
git_version = '65036c3edf13281e3c3e34e33664c9d839bff8fb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
