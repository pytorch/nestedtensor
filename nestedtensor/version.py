__version__ = '0.1.4+bcc9fab'
git_version = 'bcc9fabc9f8098b1225a5aca78a69e2ef5846836'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
