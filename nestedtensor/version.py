__version__ = '0.0.1.dev20201163+359745a'
git_version = '359745a35e33bbb77ec844b7554266d76b6340af'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
