__version__ = '0.0.1.dev202053123+b48f664'
git_version = 'b48f664fa0f1b67d1319ebba783532e1b56e5aae'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
