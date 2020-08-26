__version__ = '0.0.1.dev20208261+fec69a4'
git_version = 'fec69a40b45037ea4c3125a94c8925ec79fb77f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
