__version__ = '0.0.1.dev202041019+aed36d5'
git_version = 'aed36d50607a42109e03f0e533ba4e2d208a8272'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
