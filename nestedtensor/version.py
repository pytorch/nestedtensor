__version__ = '0.0.1.dev202052120+73cf268'
git_version = '73cf2686c300037adab40ca3a1ca11bd4df55185'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
