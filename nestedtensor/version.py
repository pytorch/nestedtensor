__version__ = '0.0.1.dev201912918+dbbb71a'
git_version = 'dbbb71a65c5c68f88df4e84e767e9e78af1da71c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
