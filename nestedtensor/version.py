__version__ = '0.0.1.dev202013119+92c7fcd'
git_version = '92c7fcd8934f8ae4995e0d40413595dc6472a2b2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
