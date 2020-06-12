__version__ = '0.0.1.dev202061216+0b50261'
git_version = '0b502619c88394a8605cb95dde41f87a8fadf1fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
