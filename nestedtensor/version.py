__version__ = '0.0.1.dev202022019+b84510e'
git_version = 'b84510e49c679a4acae368bf7b96478931e5ecb2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
