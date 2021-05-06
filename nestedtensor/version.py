__version__ = '0.1.4+8a4b182'
git_version = '8a4b18246acaf837cf1095342a01a1c951637a7d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
