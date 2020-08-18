__version__ = '0.0.1.dev20208184+2159bfc'
git_version = '2159bfc2c32507ab1a7be5e7ae10a3ebd80b9a67'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
