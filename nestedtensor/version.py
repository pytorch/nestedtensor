__version__ = '0.0.1.dev201911150+7a33f4d'
git_version = '7a33f4d291a5728195a55738a90e7773c2053a61'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
