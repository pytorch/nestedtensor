__version__ = '0.0.1.dev20201111+b94b880'
git_version = 'b94b880b892625bf6940b0ad2a4cbeb00fb95d0e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
