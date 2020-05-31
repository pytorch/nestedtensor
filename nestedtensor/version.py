__version__ = '0.0.1.dev202053117+2500fca'
git_version = '2500fcab647e225b1387e26dfe507ea547d35a8a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
