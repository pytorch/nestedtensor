__version__ = '0.1.4+3767218'
git_version = '376721869f4f9f1c80b378ca8bc02c13f46a3f49'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
