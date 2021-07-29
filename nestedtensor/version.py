__version__ = '0.1.4+eb69813'
git_version = 'eb698135840b4664a24cfe1a47b780d553fb5626'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
