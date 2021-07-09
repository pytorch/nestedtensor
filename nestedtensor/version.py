__version__ = '0.1.4+8da4a3f'
git_version = '8da4a3fe499da5c9c422d7bde287149a65e57bec'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
