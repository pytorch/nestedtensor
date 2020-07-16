__version__ = '0.0.1.dev20207162+2d2b142'
git_version = '2d2b1424325c3d9ff00f3e4a1fce86bb5ebcd559'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
