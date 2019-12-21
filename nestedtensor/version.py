__version__ = '0.0.1.dev201912211+8574dc8'
git_version = '8574dc883bb198df8a4c5cadc5fd6baf2f6c54a1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
