__version__ = '0.1.4+40b4a63'
git_version = '40b4a637ed257b0cc6dc09bd87b0508735ef4015'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
