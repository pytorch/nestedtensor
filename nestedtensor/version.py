__version__ = '0.0.1.dev202053022+1692cc9'
git_version = '1692cc97f6838ebdd03352f61f8b0f24007d790d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
