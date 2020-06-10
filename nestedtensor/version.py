__version__ = '0.0.1.dev20206103+717a9c8'
git_version = '717a9c85357ef17466b9708efb53780c17d185c1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
