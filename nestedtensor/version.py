__version__ = '0.0.1.dev2020102716+9c87812'
git_version = '9c878128f32ff1e5ff3aecee214e94ea90393183'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
