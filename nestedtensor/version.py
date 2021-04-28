__version__ = '0.0.1+8144260'
git_version = '8144260815494ff2e496818448986df6451b0dee'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
