__version__ = '0.0.1.dev202053118+1cc4f7f'
git_version = '1cc4f7f3112fc278568a5910888e32c28b10c658'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
