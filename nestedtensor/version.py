__version__ = '0.0.1.dev202022019+e9f7679'
git_version = 'e9f767961e30643c0f652fe1f75bc419b662051f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
