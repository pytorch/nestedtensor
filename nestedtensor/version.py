__version__ = '0.0.1.dev20201132+eef6de6'
git_version = 'eef6de67943ebe097f100c1c820504e268c5a5f7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
