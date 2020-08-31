__version__ = '0.0.1.dev20208314+edae05f'
git_version = 'edae05fbeb6525526e7598acd095c1ce88331548'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
