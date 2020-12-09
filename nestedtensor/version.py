__version__ = '0.0.1+753cf4e'
git_version = '753cf4ee70eef04ff6b0ab358ebe07c43648850b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
