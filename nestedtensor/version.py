__version__ = '0.1.4+26d0f06'
git_version = '26d0f06311ef74d62bc5b4fc6e32735c65945a34'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
