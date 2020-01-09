__version__ = '0.0.1.dev20201919+cf321e1'
git_version = 'cf321e1e290bd78b96ba651c8fdf850336116ac6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
