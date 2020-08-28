__version__ = '0.0.1.dev20208280+c1fd6d2'
git_version = 'c1fd6d271af9cf58929a2ab3b2a93d75ee8f88d4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
