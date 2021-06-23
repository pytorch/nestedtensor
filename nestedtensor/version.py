__version__ = '0.1.4+4137ae7'
git_version = '4137ae7c797bbdb71da4beec3c220ab1659273c7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
