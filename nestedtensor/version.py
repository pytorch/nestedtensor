__version__ = '0.0.1+b38ab7c'
git_version = 'b38ab7ce3d2e8570d03ba11698526f551e92b406'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
