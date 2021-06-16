__version__ = '0.1.4+0c7adca'
git_version = '0c7adca551f195396bf84a8ecbf5a72f7ddea32b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
