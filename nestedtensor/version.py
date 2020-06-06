__version__ = '0.0.1.dev2020665+dffae9f'
git_version = 'dffae9f697ab45f5733c90a8d976bc40f0ec1fd2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
