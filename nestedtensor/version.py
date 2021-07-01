__version__ = '0.1.4+5eb4d26'
git_version = '5eb4d26a215148d42e67e9905c5529c9e9cf876b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
