__version__ = '0.0.1.dev20203211+4c663b7'
git_version = '4c663b7bed4086cf868a6c96e56fd62b8a2dea8f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
