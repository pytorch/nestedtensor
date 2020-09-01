__version__ = '0.0.1.dev20209122+c7937fa'
git_version = 'c7937fa41f64cc1514d93dd72901e12913b5fba7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
