__version__ = '0.0.1.dev201912203+4e7555a'
git_version = '4e7555a68e89838619b88fe459947418ab02417f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
