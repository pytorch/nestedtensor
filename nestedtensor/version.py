__version__ = '0.0.1.dev20208293+5160dea'
git_version = '5160dea2a58ca629ac48fbd465b2de521a975e7c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
