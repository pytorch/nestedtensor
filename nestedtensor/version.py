__version__ = '0.0.1.dev201912135+26ea705'
git_version = '26ea705a7fd60a26d338b3b9027276e8dd7f703b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
