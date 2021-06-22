__version__ = '0.1.4+e62fffd'
git_version = 'e62fffd01980ffa18f6aea96d88c37723e3021e0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
