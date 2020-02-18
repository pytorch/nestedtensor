__version__ = '0.0.1.dev202021822+c57c778'
git_version = 'c57c7787b62d7691b2469203407511fd57846c02'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
