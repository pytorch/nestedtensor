__version__ = '0.0.1.dev202022119+956e38a'
git_version = '956e38a38e5eef9af89898e4ced5d68e424c1994'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
