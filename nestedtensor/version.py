__version__ = '0.0.1.dev201912195+98a1ebc'
git_version = '98a1ebc2f76eda00aafa03ec5e32df34f01eb1c2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
