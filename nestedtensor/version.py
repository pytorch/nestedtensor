__version__ = '0.1.4+6da549d'
git_version = '6da549d782ee8c97bc56d2e7ae6d72c2c9338180'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
