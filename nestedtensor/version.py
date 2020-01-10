__version__ = '0.0.1.dev202011023+7f813f1'
git_version = '7f813f11bb46fbcc88414c423c2d7def24837751'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
