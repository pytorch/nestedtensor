__version__ = '0.0.1.dev202021217+0449e27'
git_version = '0449e2723248eca47b0a24296e5a015df0e5946b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
