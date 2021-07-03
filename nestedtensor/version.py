__version__ = '0.1.4+4395ba9'
git_version = '4395ba91a69f8bf0a06980f522151e478143b087'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
