__version__ = '0.0.1.dev20208280+66d1c46'
git_version = '66d1c463b6461ed468fc8073c4253bf3cfd45e1a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
