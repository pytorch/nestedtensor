__version__ = '0.0.1.dev202053117+1faaa61'
git_version = '1faaa611a4794bc46ede24aec5809151b1396f9c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
