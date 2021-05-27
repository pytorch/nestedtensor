__version__ = '0.1.4+987809c'
git_version = '987809c553f44d268b6d662a9d089fc74d2802ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
