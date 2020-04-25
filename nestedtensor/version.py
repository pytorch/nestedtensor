__version__ = '0.0.1.dev20204254+2c6c987'
git_version = '2c6c987d2577f633b176c9d919a08374db21e6fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
