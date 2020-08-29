__version__ = '0.0.1.dev20208293+9a51303'
git_version = '9a51303c6c7d574c4c968d095045c9b3fd13df63'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
