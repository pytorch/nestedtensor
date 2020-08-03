__version__ = '0.0.1.dev20208321+8a59e2e'
git_version = '8a59e2ef375ef61d8099bc2d86ec14d1824ab00b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
