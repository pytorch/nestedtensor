__version__ = '0.0.1.dev20203284+0dbd56c'
git_version = '0dbd56c18c194cccd18d190ecf50d3a7678e30b6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
