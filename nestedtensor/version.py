__version__ = '0.0.1.dev202053019+1fd9c6c'
git_version = '1fd9c6c21ed054f77e9218afa67010bd9b65c552'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
