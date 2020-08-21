__version__ = '0.0.1.dev20208215+c49d05e'
git_version = 'c49d05e50e1f78cf93564ad787289ccf8618f9ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
