__version__ = '0.0.1.dev20208113+e7b2fc8'
git_version = 'e7b2fc85469c3521c7f24979135ea51296ae665b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
