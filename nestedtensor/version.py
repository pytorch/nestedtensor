__version__ = '0.0.1.dev20201172+5c12f8f'
git_version = '5c12f8f7673662a14870096839aa43214923d509'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
