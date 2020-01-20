__version__ = '0.0.1.dev20201205+e5367de'
git_version = 'e5367de4d6f200566bc93fb98bdd0fee40e95a6b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
