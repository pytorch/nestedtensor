__version__ = '0.0.1.dev20205133+6ad20c9'
git_version = '6ad20c91cba6782db6c2bd93a85c12f20cbe69b3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
