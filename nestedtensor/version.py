__version__ = '0.0.1.dev2020593+402a911'
git_version = '402a911a4c35edef4cbe196a64a8670859fa323a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
