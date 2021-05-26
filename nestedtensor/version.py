__version__ = '0.1.4+ba1afc3'
git_version = 'ba1afc3d4c35091e284f3437ab0c92b375797d54'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
