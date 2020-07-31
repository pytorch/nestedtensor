__version__ = '0.0.1.dev202073119+207df4b'
git_version = '207df4b110e659082d7f80dc338624c169721d8f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
