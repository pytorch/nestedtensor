__version__ = '0.0.1.dev20208305+368dba7'
git_version = '368dba733958210d2c933ca2cca88fcf87111b2d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
