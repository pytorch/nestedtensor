__version__ = '0.0.1.dev20207161+050937a'
git_version = '050937a4f6908eb06d69dd818b4a3577209af5a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
