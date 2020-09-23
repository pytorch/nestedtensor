__version__ = '0.0.1.dev202092316+20c5c64'
git_version = '20c5c64ace43b208c59910e10eec46d7e8ad64d9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
