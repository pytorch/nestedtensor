__version__ = '0.0.1.dev20202420+a843264'
git_version = 'a8432642d9fa47b79a1e56da0d7cf10bfd42d97a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
