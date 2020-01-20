__version__ = '0.0.1.dev20201200+02ef5f8'
git_version = '02ef5f895a8c21f5f304d625c320b60c231df984'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
