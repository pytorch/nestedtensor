__version__ = '0.0.1.dev20202194+106c11b'
git_version = '106c11b99ea7c398a13bc658466fa7cdad071572'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
