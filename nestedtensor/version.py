__version__ = '0.0.1.dev202011172+44b3563'
git_version = '44b35637049ec6babc3ee027ba44a5412b4fd46f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
