__version__ = '0.0.1.dev202052918+1317f35'
git_version = '1317f35d0810340a80ac817253d9c3d189ef5f94'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
