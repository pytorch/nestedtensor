__version__ = '0.0.1.dev20204616+8ffeb6a'
git_version = '8ffeb6ad6bbd4e5f5b8a0d4e997156b9eb98132f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
