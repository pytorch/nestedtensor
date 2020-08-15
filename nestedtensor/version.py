__version__ = '0.0.1.dev20208153+e549cfe'
git_version = 'e549cfe7d70f5146abee09adc6c3360b2f851b2c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
