__version__ = '0.0.1.dev20202196+151b556'
git_version = '151b556964fae0a4da062041edb1718444486a5d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
