__version__ = '0.0.1.dev20207192+b07a374'
git_version = 'b07a374db49bdba51349af90ee706e8db85c7ce6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
