__version__ = '0.0.1.dev20202130+232f4b9'
git_version = '232f4b959ae5d6304afc6bfbfc89bbd454aa1622'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
