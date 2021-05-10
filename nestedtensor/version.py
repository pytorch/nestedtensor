__version__ = '0.1.4+85eb327'
git_version = '85eb3271ac37186e514e06d050b5264a082f762e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
