__version__ = '0.0.1.dev20207183+98d576c'
git_version = '98d576c9caea2e0516e8b2208f2275d89b975934'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
