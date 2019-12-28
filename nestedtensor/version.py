__version__ = '0.0.1.dev201912283+0be0c80'
git_version = '0be0c807e24371703757177c3aaa6fec15075d77'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
