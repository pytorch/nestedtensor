__version__ = '0.0.1.dev202011174+ce44103'
git_version = 'ce441034e460cd18f5608e9ecf5487885ef67203'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
