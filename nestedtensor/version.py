__version__ = '0.0.1.dev20209233+51c455d'
git_version = '51c455daf8f0372b1c669685204b1462e7538326'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
