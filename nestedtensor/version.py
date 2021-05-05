__version__ = '0.1.4+77f07f2'
git_version = '77f07f22c9116b2b78dadd23570616abfcb3a727'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
