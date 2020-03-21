__version__ = '0.0.1.dev20203211+eb02133'
git_version = 'eb0213333104096cab34300b45a570dffa1959a5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
