__version__ = '0.0.1.dev202053121+723c552'
git_version = '723c55213d756ac9769692a7da8f75c282bb2afa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
