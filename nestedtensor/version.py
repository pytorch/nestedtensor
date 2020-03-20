__version__ = '0.0.1.dev202032018+463f049'
git_version = '463f049e2a116606a94ac7a45849888c9db7f584'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
