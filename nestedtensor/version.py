__version__ = '0.0.1.dev20205301+57b95bd'
git_version = '57b95bd5901283280613fd36aec8860295fcb3dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
