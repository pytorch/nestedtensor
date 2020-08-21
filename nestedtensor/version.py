__version__ = '0.0.1.dev202082114+8c5171f'
git_version = '8c5171f548b64bc5f70b3902dbd22f7667983be5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
