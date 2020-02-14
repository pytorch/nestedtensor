__version__ = '0.0.1.dev202021420+6b9cf36'
git_version = '6b9cf36219f56d565f5cdedb716e69331f110925'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
