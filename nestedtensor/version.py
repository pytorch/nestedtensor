__version__ = '0.0.1.dev20208296+03ffe10'
git_version = '03ffe1006fd788f7d281754bdb49145969bb4761'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
