__version__ = '0.0.1.dev20208215+5ee847d'
git_version = '5ee847d8cf4fbb20738d4b46bf265ac351b7774b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
