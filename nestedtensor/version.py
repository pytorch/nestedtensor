__version__ = '0.0.1.dev20201166+e67dbef'
git_version = 'e67dbeff2b16ca772fb83ce8f2f4d178283a6bb9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
