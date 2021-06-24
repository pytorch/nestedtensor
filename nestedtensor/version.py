__version__ = '0.1.4+b473f91'
git_version = 'b473f9157710fa3ab255c2bed6eefee8e9a2591f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
