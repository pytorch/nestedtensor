__version__ = '0.0.1.dev20202221+6f17108'
git_version = '6f17108a0c1971e97f8176007411cb4a69556c48'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
