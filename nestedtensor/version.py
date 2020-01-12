__version__ = '0.0.1.dev20201122+00d5796'
git_version = '00d579661b2e93046c7666d752ca8e0e063e2f9d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
