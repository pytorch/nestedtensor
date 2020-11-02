__version__ = '0.0.1.dev202011222+8610349'
git_version = '8610349882df4a625281521cd6f657a0ef94db8e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
