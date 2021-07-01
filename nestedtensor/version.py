__version__ = '0.1.4+88e2d4b'
git_version = '88e2d4b69734365f0e541841fb30f38e51fdc22d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
