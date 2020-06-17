__version__ = '0.0.1.dev20206172+8ca4696'
git_version = '8ca4696fd71d6afc444b477ebfdbb0a361fc11c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
