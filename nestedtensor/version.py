__version__ = '0.0.1.dev20204420+4b9a7bc'
git_version = '4b9a7bce1754dd35648a78689c2ae890c563842d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
