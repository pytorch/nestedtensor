__version__ = '0.0.1.dev2019121420+a5dfdf7'
git_version = 'a5dfdf783b4186371112e56ed9addac5dc98117e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
