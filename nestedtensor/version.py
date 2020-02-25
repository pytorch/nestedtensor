__version__ = '0.0.1.dev20202252+7c94f9f'
git_version = '7c94f9f6675add46b6fa84fd7fd602a1bc5c81e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
