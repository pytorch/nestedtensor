__version__ = '0.1.4+8fe9739'
git_version = '8fe9739faefc9ad067232162951e2c14eca7b506'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
