__version__ = '0.0.1.dev202081922+8d63e2f'
git_version = '8d63e2fb3ef82735cfeff80101dbd1d1e407475d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
