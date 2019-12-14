__version__ = '0.0.1.dev2019121421+45c3fe8'
git_version = '45c3fe8145627d9bda80f58b226e4ab0c03bffbb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
