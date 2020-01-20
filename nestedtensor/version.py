__version__ = '0.0.1.dev202012021+bde75ce'
git_version = 'bde75ce1a064aa35f73326a0d4d499357a1260c1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
