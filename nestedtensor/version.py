__version__ = '0.0.1.dev202072319+42441c1'
git_version = '42441c130b0ac61fdfbdf10a0c727a1fa0bf7862'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
