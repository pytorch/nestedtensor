__version__ = '0.1.4+bb543a9'
git_version = 'bb543a98ff71c6aad1b9cadc458239a853053015'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
