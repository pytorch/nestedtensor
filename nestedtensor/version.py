__version__ = '0.0.1.dev20203274+2fd059f'
git_version = '2fd059fc6f51e384af1bf66280ee5bd6bdad76cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
