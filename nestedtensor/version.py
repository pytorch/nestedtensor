__version__ = '0.0.1.dev20207153+238194f'
git_version = '238194f9aba3866679634c3ab1320153c0c98f4f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
