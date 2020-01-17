__version__ = '0.0.1.dev20201174+7b5416d'
git_version = '7b5416daaef9918e733bc693699b233c9883b1b2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
