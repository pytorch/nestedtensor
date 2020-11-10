__version__ = '0.0.1.dev202011100+a413b70'
git_version = 'a413b7060761678d879ecbfc1fd01e42d43285f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
