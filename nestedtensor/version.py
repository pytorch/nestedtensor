__version__ = '0.0.1.dev202071520+33a4f57'
git_version = '33a4f570bca8d603f62c8cb7da854a3e6dd2531d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
