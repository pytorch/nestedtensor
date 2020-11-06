__version__ = '0.0.1.dev202011517+fbaacf1'
git_version = 'fbaacf1035ecb62cddeb8572ba7f5def318ea887'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
