__version__ = '0.0.1.dev201912264+bea2e46'
git_version = 'bea2e46df08f2c06375155e1a482db88591ee964'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
