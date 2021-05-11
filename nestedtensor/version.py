__version__ = '0.1.4+950360d'
git_version = '950360d305df1147a71615edcc9428dd72791579'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
