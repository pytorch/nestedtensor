__version__ = '0.0.1.dev20207110+676dafa'
git_version = '676dafab8bd8d4df2d3fceb1a8c2b89bfe78471d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
