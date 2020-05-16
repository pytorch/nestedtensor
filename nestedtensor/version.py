__version__ = '0.0.1.dev20205161+81ae0e3'
git_version = '81ae0e3c3b34347033a736ca43d72546c7a38495'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
