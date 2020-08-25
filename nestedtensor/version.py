__version__ = '0.0.1.dev20208254+8c381e2'
git_version = '8c381e21692b923eb2ba58b84c1fe5955ae207ad'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
