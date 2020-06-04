__version__ = '0.0.1.dev2020643+aef6c81'
git_version = 'aef6c81a69365e475c8c58139079cd3ae9863b4f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
