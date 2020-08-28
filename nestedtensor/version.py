__version__ = '0.0.1.dev20208283+0c2cf5d'
git_version = '0c2cf5dbe8bf17d1b5c5f62fdaded75d6004aa54'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
