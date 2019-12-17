__version__ = '0.0.1.dev201912171+ddcee83'
git_version = 'ddcee83837d2bea240224e272cbd09d803a52f36'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
