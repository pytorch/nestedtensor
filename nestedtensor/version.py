__version__ = '0.0.1.dev2019121422+7a50534'
git_version = '7a505342fb5eb919b2960a98757d11bf860485cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
