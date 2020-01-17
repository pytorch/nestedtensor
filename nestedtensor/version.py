__version__ = '0.0.1.dev20201171+8e10f3e'
git_version = '8e10f3eece951d8103b2d746702a057815e441ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
