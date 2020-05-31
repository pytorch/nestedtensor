__version__ = '0.0.1.dev202053118+59b1d24'
git_version = '59b1d243a6b0a3eb2c4ae0bdeea953d53cf94b50'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
