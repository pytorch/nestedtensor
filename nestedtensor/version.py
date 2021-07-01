__version__ = '0.1.4+e671bee'
git_version = 'e671bee1aca77f9f825dceed6c4b89c9b835cac7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
