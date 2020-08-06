__version__ = '0.0.1.dev20208618+42223b7'
git_version = '42223b7da646516d5da37c73a939a9f4f18c826d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
