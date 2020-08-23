__version__ = '0.0.1.dev20208236+045791d'
git_version = '045791d0b6d674388bcefdf32806675c6c98a248'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
