__version__ = '0.0.1.dev202022115+d268bc8'
git_version = 'd268bc83ef16c7cd59204673adbd40d3f41fe78f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
