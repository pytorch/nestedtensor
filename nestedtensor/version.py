__version__ = '0.0.1.dev202022122+dd6334c'
git_version = 'dd6334c68e1c59e8aaffbac87f01407ef1264685'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
