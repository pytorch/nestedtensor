__version__ = '0.0.1.dev20208112+fc56fb4'
git_version = 'fc56fb4b2d921d843ff9e49f8e8b98fbe8e6ea3a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
