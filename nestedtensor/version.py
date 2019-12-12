__version__ = '0.0.1.dev2019121223+6695c05'
git_version = '6695c05dc8551f66bb512b6b4e26451782c46a7d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
