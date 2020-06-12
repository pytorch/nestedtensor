__version__ = '0.0.1.dev202061220+b621d27'
git_version = 'b621d27293b1854af36bbf3363f6e79b72aa360c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
