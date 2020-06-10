__version__ = '0.0.1.dev20206102+fbdcb6e'
git_version = 'fbdcb6e057f57f1654f0c68b45d2ee57f592c0bd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
