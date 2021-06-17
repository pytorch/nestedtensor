__version__ = '0.1.4+694c388'
git_version = '694c388ed4d270a8d131b2a0575c36e1609f2f67'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
