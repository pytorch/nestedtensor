__version__ = '0.0.1.dev2020642+474b0cc'
git_version = '474b0cc2b8b6256ffbe553fecd463106cc2625bc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
