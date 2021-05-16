__version__ = '0.1.4+6ab3e4f'
git_version = '6ab3e4f86ac9ae9ec177173e76c50c36da60f4d1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
