__version__ = '0.0.1.dev202053118+516a762'
git_version = '516a762d21d09b42ba85acad9fe3def899f241dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
