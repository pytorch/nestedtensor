__version__ = '0.1.4+710cc81'
git_version = '710cc81b51fb55485a3c2c0404317169b889fa8d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
