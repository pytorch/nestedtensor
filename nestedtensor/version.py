__version__ = '0.0.1.dev20205192+0c92324'
git_version = '0c923241ce5220cd84cec4b3b1182a3ddcffe03d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
