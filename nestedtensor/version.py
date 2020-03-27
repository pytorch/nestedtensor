__version__ = '0.0.1.dev20203274+d1008b7'
git_version = 'd1008b728fe599b547cedc80f6c3079c10206624'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
