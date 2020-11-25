__version__ = '0.0.1.dev2020112521+51ff306'
git_version = '51ff306d11354a1785787f13623b4e2048bca617'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
