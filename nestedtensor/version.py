__version__ = '0.1.4+9ba11d6'
git_version = '9ba11d6f7491ac0b99e5ca627024e32192358674'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
