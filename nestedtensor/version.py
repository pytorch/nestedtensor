__version__ = '0.1.4+986cfd5'
git_version = '986cfd55e2d0c8139a5e19cfca6efc740ea7ad23'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
