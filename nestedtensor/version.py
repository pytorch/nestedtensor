__version__ = '0.0.1.dev202052122+b187c45'
git_version = 'b187c45c1b5d060c25349aa3adf5ab18cde6258c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
