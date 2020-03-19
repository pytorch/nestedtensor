<<<<<<< HEAD
__version__ = '0.0.1.dev202032921+bc62b71'
git_version = 'bc62b7119bc0fb29511cf54699b69ce9724f02bb'
=======
__version__ = '0.0.1.dev202031816+955d513'
git_version = '955d5138490c9cb16d0b3cbad6fa444a30ecb31c'
>>>>>>> Set up ShipIt
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
