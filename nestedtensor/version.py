__version__ = '0.0.1.dev20207161+8473cb1'
git_version = '8473cb1d2010e2ab28b90a2c391d559c59b69e01'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
