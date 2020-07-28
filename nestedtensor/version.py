__version__ = '0.0.1.dev202072815+ef93d5b'
git_version = 'ef93d5b919cd32ae8973fddeed2845ca5635370f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
