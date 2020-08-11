__version__ = '0.0.1.dev20208112+3ddcfed'
git_version = '3ddcfed9fa53287fff772059b2e550ce50dda552'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
