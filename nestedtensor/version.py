__version__ = '0.0.1.dev201912719+2605579'
git_version = '260557967c809623bfb48ab636b7ccec401760d5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
