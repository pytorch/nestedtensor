__version__ = '0.1.4+b09167f'
git_version = 'b09167fe118c66c5d2dad2befa0d954d74916268'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
