__version__ = '0.0.1.dev202011520+832b078'
git_version = '832b0780a5d3a865cf0c938c404b182e8b9247e8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
