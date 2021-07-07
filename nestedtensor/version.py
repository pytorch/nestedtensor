__version__ = '0.1.4+b5b293f'
git_version = 'b5b293f91bb12a77d7031d9242651bc4a5158e63'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
