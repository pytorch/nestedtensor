__version__ = '0.0.1+5fb33cf'
git_version = '5fb33cfc699e7d474d9ae99b45f35f01cd89bd81'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_C_EXTENSION=True
