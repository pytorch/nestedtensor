__version__ = '0.1.4+45e75ce'
git_version = '45e75ce1d1d28b24122979652f386763ac7ff2a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
