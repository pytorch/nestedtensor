__version__ = '0.0.1.dev20202254+a8d5490'
git_version = 'a8d54909e4104e817701ec1b4fd6c79a3a793995'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
