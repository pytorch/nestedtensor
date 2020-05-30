__version__ = '0.0.1.dev202053019+af5caa5'
git_version = 'af5caa5fa9f629734a8a945a8eb7cac474f48da3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
