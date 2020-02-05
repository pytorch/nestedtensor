__version__ = '0.0.1.dev2020252+00977b5'
git_version = '00977b532bd04376dc1082d4613323c0a2d5e24b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
