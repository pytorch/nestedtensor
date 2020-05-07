__version__ = '0.0.1.dev20205417+6a6dd44'
git_version = '6a6dd445d9931f1618269d8e11fb540e9b01ac45'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
