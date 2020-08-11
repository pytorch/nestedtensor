__version__ = '0.0.1.dev20208113+4d3b70e'
git_version = '4d3b70e4073ad20627cb1aa22ae5a7b6a8dbca0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
