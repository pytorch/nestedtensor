__version__ = '0.0.1.dev202053019+1a34dc0'
git_version = '1a34dc0ac08c9efad2889d0cc0b56545f645b99a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
