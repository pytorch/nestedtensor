__version__ = '0.1.4+9cc25a1'
git_version = '9cc25a17976d092be46b5134e150ad76f331db00'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
