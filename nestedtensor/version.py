__version__ = '0.0.1.dev20209114+46f958d'
git_version = '46f958d22011ae5ccfc538915db4ce2277d3f189'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
