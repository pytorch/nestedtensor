__version__ = '0.0.1.dev202081922+bf1c67a'
git_version = 'bf1c67a94c863fedb0ea6b5a9a5577f5d3507113'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
