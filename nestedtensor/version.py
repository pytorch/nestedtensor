__version__ = '0.1.4+31b5303'
git_version = '31b5303389f76f9588f429b23de48129acbee4c1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
