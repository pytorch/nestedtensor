__version__ = '0.0.1.dev20202196+4de62bc'
git_version = '4de62bc6c5c2f7e46f61fec85f471c34b85e201f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
