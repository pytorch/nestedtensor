__version__ = '0.0.1.dev20202194+2a3a715'
git_version = '2a3a715b5941835e4ba2ef3c85221ac6d2abf728'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
