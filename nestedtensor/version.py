__version__ = '0.1.4+a8ad916'
git_version = 'a8ad916f79676b826817e99650dfc010a7842091'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
