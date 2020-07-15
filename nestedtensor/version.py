<<<<<<< HEAD
__version__ = '0.0.1.dev202071519+bff2ba7'
git_version = 'bff2ba75d2d1c1139d1d35a19dc450c9b652ae7f'
=======
__version__ = '0.0.1.dev202071515+2fb94d8'
git_version = '2fb94d8d788650f4bf4340988b0a2c0a3684fbe2'
>>>>>>> master
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
