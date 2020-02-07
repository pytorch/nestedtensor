__version__ = '0.0.1.dev2020276+5d2e264'
git_version = '5d2e2645cba7238926f2f596dccd44af3e96495f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
