__version__ = '0.0.1.dev20208301+23e7261'
git_version = '23e72610af647947e84ebaec34ca181af0c68a7f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
