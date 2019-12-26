__version__ = '0.0.1.dev2019122622+a71bd21'
git_version = 'a71bd21ca234bae31f345169351c70ea4174460d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
