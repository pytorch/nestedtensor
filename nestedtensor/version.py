__version__ = '0.0.1.dev20209250+439102b'
git_version = '439102b131b3487b07a2f35fdc0362566420f944'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
