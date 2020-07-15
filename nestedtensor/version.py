__version__ = '0.0.1.dev20207153+695d382'
git_version = '695d38299ab44ca57f7e1ea7385de22f37d886f5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
