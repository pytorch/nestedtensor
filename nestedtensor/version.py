__version__ = '0.0.1.dev2020112416+db5c11f'
git_version = 'db5c11f0337ab10069711c38d75284d607e249c4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
