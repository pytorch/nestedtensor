__version__ = '0.0.1.dev202011618+52e9815'
git_version = '52e9815e72e927b380198d3d4a8e564d1e882c40'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
