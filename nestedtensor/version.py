__version__ = '0.0.1.dev20201204+c3f8b1d'
git_version = 'c3f8b1dd290bbeaa99a5964344a792627b4d2b3e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
