__version__ = '0.1.3'
git_version = 'b1f2ba969c1485041a45f30617270647b3cdca9d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
