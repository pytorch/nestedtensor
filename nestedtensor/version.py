__version__ = '0.0.1.dev202041018+2074ab8'
git_version = '2074ab8e7ade3c4a963d30065217d15046017fa1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
