__version__ = '0.0.1+8499dfe'
git_version = '8499dfe9a4101ac42e51e96eaa9112d75ec6fe53'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
