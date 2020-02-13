__version__ = '0.0.1.dev20202130+cbf1bc9'
git_version = 'cbf1bc9710fe7b893aa129033026d64ae33438f5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
