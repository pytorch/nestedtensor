__version__ = '0.0.1.dev2020111121+e464008'
git_version = 'e464008aed6b3d264b40a6ffb1e48bbc57093884'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
