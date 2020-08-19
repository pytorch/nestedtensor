__version__ = '0.0.1.dev20208195+e67cab7'
git_version = 'e67cab79059fc0b45db4bf275171f01db1f9aa41'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
