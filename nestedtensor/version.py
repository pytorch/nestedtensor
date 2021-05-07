__version__ = '0.1.4+d958933'
git_version = 'd9589331ac04ba0a29ff36acf27b70df4a3a8624'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
