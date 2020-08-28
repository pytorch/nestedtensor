__version__ = '0.0.1.dev20208280+2639ffb'
git_version = '2639ffb4728c1cd0079d0c0d0016ad51e40fad76'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
