__version__ = '0.1.4+0a160d8'
git_version = '0a160d85c20d0e74217b7427c583d4a3db4f7933'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
