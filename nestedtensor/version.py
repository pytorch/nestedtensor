__version__ = '0.0.1.dev202081413+9fbd9b3'
git_version = '9fbd9b39ca120fdb03268c130938b96c63d222fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
