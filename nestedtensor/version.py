__version__ = '0.0.1.dev20207162+48e8ce8'
git_version = '48e8ce85060310afbc979df2137aa15a04916d23'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
