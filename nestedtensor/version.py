__version__ = '0.0.1.dev20207153+75d72e2'
git_version = '75d72e27e3ce592c294b3621eb72911c642103f1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
