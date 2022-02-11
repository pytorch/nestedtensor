__version__ = '0.1.4+42e79bb'
git_version = '42e79bb0f7a57e0203cd77af7f6c88c5fdc0ff9a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
