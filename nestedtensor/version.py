__version__ = '0.0.1.dev202012021+dfd465f'
git_version = 'dfd465f43600c4b114b334fcbd75546c4bd010c3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
