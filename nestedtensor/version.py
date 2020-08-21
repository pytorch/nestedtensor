__version__ = '0.0.1.dev202082119+07382a6'
git_version = '07382a61f08cecd5589fe499266c7acba939df0f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
