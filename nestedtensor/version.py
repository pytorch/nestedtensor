__version__ = '0.0.1.dev2019121323+edceb18'
git_version = 'edceb18685da80319285bf34f6019fd79bda59b2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
