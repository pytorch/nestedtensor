__version__ = '0.0.1.dev20208203+4037c2e'
git_version = '4037c2ef2edb14286f15cf140042a7cba06530a5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
