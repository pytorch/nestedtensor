__version__ = '0.0.1.dev20201212+5bec91d'
git_version = '5bec91dde52a5af2cfec19a1a13e2d8eda0cce9c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
