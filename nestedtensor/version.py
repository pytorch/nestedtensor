__version__ = '0.1.4+be9f4cd'
git_version = 'be9f4cdab5afbc377e16ef030a3ca937af524ad5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
