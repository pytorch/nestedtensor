__version__ = '0.0.1.dev202012322+cd4a74b'
git_version = 'cd4a74b0190d2318e6d717a9113d033c6a14a2bd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
