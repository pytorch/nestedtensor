__version__ = '0.0.1.dev20202195+0e3e986'
git_version = '0e3e98673f2bea2b6c0f8aaecef9513cce161080'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
