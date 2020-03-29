__version__ = '0.0.1.dev20203293+bb3914f'
git_version = 'bb3914feb40a3d45d72db55d924b3c10595ddfa6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
