__version__ = '0.0.1.dev20201301+f30bd96'
git_version = 'f30bd96beeda1eb3d9da25385560167be80ebd5f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
