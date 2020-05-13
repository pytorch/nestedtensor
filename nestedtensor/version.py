__version__ = '0.0.1.dev202051322+1e35175'
git_version = '1e35175411743d8f4cd85d55504d2038c10b0dbc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
