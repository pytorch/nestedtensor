__version__ = '0.1.4+85a6da9'
git_version = '85a6da98c4128ebe3ce4cbc43eae654998f7a62b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
