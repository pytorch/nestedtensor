__version__ = '0.0.1.dev20208154+7fd039d'
git_version = '7fd039d37fea7525d4fa495bfce5f1d787e62a0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
