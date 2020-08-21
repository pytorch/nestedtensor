__version__ = '0.0.1.dev202082122+f17586d'
git_version = 'f17586d0f7325d2743888ec6e5b7b9854ac931c1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
