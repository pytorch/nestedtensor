__version__ = '0.0.1.dev20206171+910e9d4'
git_version = '910e9d4629c4789e4d90debf655f154b4743f7bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
