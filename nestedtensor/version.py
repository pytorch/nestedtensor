__version__ = '0.1.4+bf12d17'
git_version = 'bf12d17c3b7891c713cb16b7e36926b873813ceb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
