__version__ = '0.0.1.dev202022721+2f4d7a2'
git_version = '2f4d7a29167241e0633042ca7136d3d032e9cd49'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
